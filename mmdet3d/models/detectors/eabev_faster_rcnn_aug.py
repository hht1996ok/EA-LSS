import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np

from mmdet.models import DETECTORS
from mmdet3d.models.detectors import MVXFasterRCNN
from .cam_stream_lss import LiftSplatShoot
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from torchvision.utils import save_image
from mmcv.cnn import ConvModule, xavier_init
import torch.nn as nn
class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

@DETECTORS.register_module()
class EABEV_FasterRCNN_Aug(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(self, lss=False, lc_fusion=False, camera_stream=False,
                camera_depth_range=[4.0, 45.0, 1.0], img_depth_loss_weight=1.0,  img_depth_loss_method='kld',
                grid=0.6, num_views=6, se=False,
                final_dim=(900, 1600), pc_range=[-50, -50, -5, 50, 50, 3], downsample=4, imc=256, lic=384, step=7,  **kwargs):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            lc_fusion (bool): fusing multi-modalities of camera and LiDAR in BEVFusion.
            camera_stream (bool): using camera stream.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            grid, num_views, final_dim, pc_range, downsample: args for LSS, see cam_stream_lss.py.
            imc (int): channel dimension of camera BEV feature.
            lic (int): channel dimension of LiDAR BEV feature.

        """
        super(EABEV_FasterRCNN_Aug, self).__init__(**kwargs)
        self.num_views = num_views
        self.lc_fusion = lc_fusion
        self.img_depth_loss_weight = img_depth_loss_weight
        self.img_depth_loss_method = img_depth_loss_method
        self.camera_depth_range = camera_depth_range
        self.lift = camera_stream
        self.step = step
        self.se = se
        if camera_stream:
            self.lift_splat_shot_vis = LiftSplatShoot(lss=lss, grid=grid, inputC=imc, camC=64, 
            pc_range=pc_range, final_dim=final_dim, downsample=downsample)
        if lc_fusion:
            if se:
                self.seblock = SE_Block(lic)
            self.reduc_conv = ConvModule(
                lic + imc,
                lic,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)
            
        self.freeze_img = kwargs.get('freeze_img', False)
        self.init_weights(pretrained=kwargs.get('pretrained', None))
        self.freeze()

    def freeze(self):
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            if self.lift:
                for param in self.lift_splat_shot_vis.parameters():
                    param.requires_grad = False


    def extract_pts_feat(self, pts, img_feats, img_metas, gt_bboxes_3d=None):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)


        x = self.pts_backbone(x)

        if self.with_pts_neck:
            x = self.pts_neck(x)

        return x
    
    
    def extract_feat(self, points, img, img_metas, img_aug_matrix=None, lidar_aug_matrix=None, gt_bboxes_3d=None):
        """Extract features from images and points."""
        img_size = img.size()
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)

        if self.lift:
            BN, C, H, W = img_feats[0].shape
            batch_size = BN//self.num_views
            img_feats_view = img_feats[0].view(batch_size, self.num_views, C, H, W)
            rots = []
            trans = []
            for sample_idx in range(batch_size):
                rot_list = []
                trans_list = []
                for mat in img_metas[sample_idx]['lidar2img']:  
                    mat = torch.Tensor(mat).to(img_feats_view.device)
                    rot_list.append(mat.inverse()[:3, :3])
                    trans_list.append(mat.inverse()[:3, 3].view(-1))

                rot_list = torch.stack(rot_list, dim=0)
                trans_list = torch.stack(trans_list, dim=0)
                rots.append(rot_list)
                trans.append(trans_list)
            rots = torch.stack(rots)
            trans = torch.stack(trans)
            lidar2img_rt = img_metas[sample_idx]['lidar2img']

            post_rots=None
            post_trans=None
            if img_aug_matrix is not None:
                img_aug_matrix = torch.stack(img_aug_matrix).permute(1, 0, 2, 3)
                post_rots = img_aug_matrix[..., :3, :3]
                post_trans = img_aug_matrix[..., :3, 3]
            
            extra_rots=None
            extra_trans=None
            if lidar_aug_matrix is not None:
                lidar_aug_matrix = lidar_aug_matrix.unsqueeze(1).repeat(1, self.num_views, 1, 1)
                extra_rots = lidar_aug_matrix[..., :3, :3]
                extra_trans = lidar_aug_matrix[..., :3, 3]

            batch_size = len(points)
            depth = torch.zeros(batch_size, img_size[1], 1, img_size[3], img_size[4]).cuda()
            for b in range(batch_size):
                cur_coords = points[b].float()[:, :3]
                if lidar_aug_matrix is not None:
                    cur_lidar_aug_matrix = lidar_aug_matrix[b][0]
                    cur_coords -= cur_lidar_aug_matrix[:3, 3]
                    cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                        cur_coords.transpose(1, 0)
                    )
                else:
                    cur_coords = cur_coords.transpose(1, 0)

                cur_lidar2image = []
                for lidar2image in lidar2img_rt:
                    cur_lidar2image.append(torch.tensor(lidar2image).unsqueeze(0))
                cur_lidar2image = torch.cat(cur_lidar2image, dim=0).cuda().float()
                cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
                dist = cur_coords[:, 2, :]
                cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-4, 1e4)
                cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

                if img_aug_matrix is not None:
                    cur_img_aug_matrix = img_aug_matrix[b]
                    cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
                    cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
                cur_coords = cur_coords[:, :2, :].transpose(1, 2)

                cur_coords = cur_coords[..., [1, 0]]
                on_img = (
                        (cur_coords[..., 0] < img_size[3])
                        & (cur_coords[..., 0] >= 0)
                        & (cur_coords[..., 1] < img_size[4])
                        & (cur_coords[..., 1] >= 0)
                )
                for c in range(on_img.shape[0]):
                    masked_coords = cur_coords[c, on_img[c]].long()
                    masked_dist = dist[c, on_img[c]]
                    depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist.float()

            self.step = int(self.step)
            B, N, C, H, W = depth.size()
            depth_tmp = depth.reshape(B * N, C, H, W)
            pad = int((self.step - 1) // 2)
            depth_tmp = F.pad(depth_tmp, [pad, pad, pad, pad], mode='constant', value=0)
            patches = depth_tmp.unfold(dimension=2, size=self.step, step=1)
            patches = patches.unfold(dimension=3, size=self.step, step=1)
            max_depth, _ = patches.reshape(B, N, C, H, W, -1).max(dim=-1)
            img_metas[0].update({'max_depth': max_depth})

            self.step = float(self.step)
            shift_list = [[self.step / H, 0.0 / W], [-self.step / H, 0.0 / W], [0.0 / H, self.step / W],
                          [0.0 / H, -self.step / W]]
            max_depth_tmp = max_depth.reshape(B * N, C, H, W)
            output_list = []
            for shift in shift_list:
                transform_matrix = torch.tensor([[1, 0, shift[0]], [0, 1, shift[1]]]).unsqueeze(0).repeat(B * N, 1,
                                                                                                          1).cuda()
                grid = F.affine_grid(transform_matrix, max_depth_tmp.shape).float()
                output = F.grid_sample(max_depth_tmp, grid, mode='nearest').reshape(B, N, C, H, W)
                output = max_depth - output
                output_mask = ((output == max_depth) == False)
                output = output * output_mask
                output_list.append(output)
            grad = torch.cat(output_list, dim=2)
            max_grad = torch.abs(grad).max(dim=2)[0].unsqueeze(2)
            img_metas[0].update({'max_grad': max_grad})
            depth = torch.cat([depth, grad], dim=2)

            img_bev_feat, img_metas = self.lift_splat_shot_vis(img_feats_view, rots, trans, lidar2img_rt=lidar2img_rt, img_metas=img_metas,
                                                                post_rots=post_rots, post_trans=post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
                                                                depth_lidar=depth)
            

            if pts_feats is None:
                pts_feats = [img_bev_feat] ####cam stream only
            else:
                if self.lc_fusion:
                    pts_feats = [self.reduc_conv(torch.cat([img_bev_feat, pts_feats[0]], dim=1))]
                    if self.se:
                        pts_feats = [self.seblock(pts_feats[0])]
        return dict(
            img_feats = img_feats,
            pts_feats = pts_feats,
            img_metas = img_metas
        )
    
    def simple_test(self, points, img_metas, img=None, img_aug_matrix=None, rescale=False):
        """Test function without augmentaiton."""
        if img_aug_matrix is not None:
            img_aug_matrix = img_aug_matrix[0]
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas, img_aug_matrix=img_aug_matrix)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                # pts_feats, img_feats, img_metas, rescale=rescale)
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      img_aug_matrix=None,
                      lidar_aug_matrix=None,
                      proposals=None,
                      gt_bboxes_ignore=None):


        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, img_aug_matrix=img_aug_matrix, lidar_aug_matrix=lidar_aug_matrix)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats'] 
        img_metas = feature_dict['img_metas']

        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

