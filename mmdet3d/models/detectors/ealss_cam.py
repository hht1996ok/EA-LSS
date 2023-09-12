import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmcv.cnn import ConvModule, xavier_init
from mmdet.models import DETECTORS
from .. import builder

from mmdet3d.models.detectors import MVXFasterRCNN
from .cam_stream_lss import LiftSplatShoot

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
class EALSS_CAM(MVXFasterRCNN):
    """Base class of Multi-modality VoxelNet."""
    def __init__(self, lss=False, lc_fusion=False, camera_stream=False,
                camera_depth_range=[4.0, 45.0, 1.0], img_depth_loss_weight=1.0,  img_depth_loss_method='kld',
                grid=0.6, num_views=6, se=False, final_dim=(900, 1600), pc_range=[-50, -50, -5, 50, 50, 3],
                downsample=4, imc=512, lic=384, step=7, **kwargs):

        super(EALSS_CAM, self).__init__(**kwargs)
        self.num_views = num_views
        self.lc_fusion = lc_fusion
        self.img_depth_loss_weight = img_depth_loss_weight
        self.img_depth_loss_method = img_depth_loss_method
        self.camera_depth_range = camera_depth_range
        self.lift = camera_stream
        self.step = step
        self.final_dim = final_dim
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
        self.init_weights_img(pretrained=kwargs.get('pretrained', None))
        self.freeze()

    def init_weights_img(self, pretrained=None):
        """Initialize model weights."""
        super(EALSS_CAM, self).init_weights(pretrained)
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

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

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img.float())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        # if not self.with_pts_bbox:
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

    def extract_feat(self, points, img, img_metas, gt_bboxes_3d=None):
        """Extract features from images and points."""
        img_size = img.size()  # 1, 6, 3, 448, 800
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)

        if self.lift:
            BN, C, H, W = img_feats[0].shape
            batch_size = BN // self.num_views
            img_feats_view = img_feats[0].view(batch_size, self.num_views, C, H, W)
            rots = []
            trans = []
            rots_depth = []
            trans_depth = []
            for sample_idx in range(batch_size):
                rot_list = []
                trans_list = []
                rot_depth_list = []
                trans_depth_list = []
                for mat in img_metas[sample_idx]['lidar2img']:
                    mat = torch.Tensor(mat)
                    rot_list.append(mat.inverse()[:3, :3].to(img_feats_view.device))
                    trans_list.append(mat.inverse()[:3, 3].view(-1).to(img_feats_view.device))
                    rot_depth_list.append(mat[:3, :3].to(img_feats_view.device))
                    trans_depth_list.append(mat[:3, 3].view(-1).to(img_feats_view.device))
                rot_list = torch.stack(rot_list, dim=0)
                trans_list = torch.stack(trans_list, dim=0)
                rot_depth_list = torch.stack(rot_depth_list, dim=0)
                trans_depth_list = torch.stack(trans_depth_list, dim=0)
                rots.append(rot_list)
                trans.append(trans_list)
                rots_depth.append(rot_depth_list)
                trans_depth.append(trans_depth_list)
            rots = torch.stack(rots)
            trans = torch.stack(trans)
            rots_depth = torch.stack(rots_depth)  # depth transform 4 6 3 3
            trans_depth = torch.stack(trans_depth)  # depth transform
            lidar2img_rt = img_metas[sample_idx]['lidar2img']  #### extrinsic parameters for multi-view images

            batch_size = len(points)
            depth = torch.zeros(batch_size, img_size[1], 1, img_size[3], img_size[4]).cuda() # 创建大小
            for b in range(batch_size):
                cur_coords = points[b].float()[:, :3]  #取点的xyz
                # lidar2image
                cur_coords = rots_depth[b].matmul(cur_coords.transpose(1, 0))
                cur_coords += trans_depth[b].reshape(-1, 3, 1)

                # get 2d coords
                dist = cur_coords[:, 2, :]
                cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-4, 1e4)
                cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
                # imgaug

                cur_coords = cur_coords[:, :2, :].transpose(1, 2)
                # normalize coords for grid sample
                cur_coords = cur_coords[..., [1, 0]]

                on_img = (
                    (cur_coords[..., 0] < img_size[3])
                    & (cur_coords[..., 0] >= 0)
                    & (cur_coords[..., 1] < img_size[4])
                    & (cur_coords[..., 1] >= 0)
                )

                for c in range(on_img.shape[0]):
                    masked_coords = cur_coords[c, on_img[c]].long()  # 点云投影到图像坐标
                    masked_dist = dist[c, on_img[c]]  # 对应深度
                    depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist.float()  # 稀疏的深度约束图（用于计算loss） 1, 6, 1, 448, 800

            step = 7
            B, N, C, H, W = depth.size()
            depth_tmp = depth.reshape(B*N, C, H, W)
            pad = int((step - 1) // 2)
            depth_tmp = F.pad(depth_tmp, [pad, pad, pad, pad], mode='constant', value=0)
            patches = depth_tmp.unfold(dimension=2, size=step, step=1)
            patches = patches.unfold(dimension=3, size=step, step=1)
            max_depth, _ = patches.reshape(B, N, C, H, W, -1).max(dim=-1)  # [2, 6, 1, 256, 704]
            img_metas[0].update({'max_depth': max_depth})

            # 求解max_depth四个方向梯度, 随后concat depth, 以缓解深度跳变对深度预测模块的影响
            step = float(step)
            shift_list = [[step / H, 0.0 / W], [-step / H, 0.0 / W], [0.0 / H, step / W], [0.0 / H, -step / W]]
            max_depth_tmp = max_depth.reshape(B*N, C, H, W)
            output_list = []
            for shift in shift_list:
                transform_matrix =torch.tensor([[1, 0, shift[0]],[0, 1, shift[1]]]).unsqueeze(0).repeat(B*N, 1, 1).cuda()
                grid = F.affine_grid(transform_matrix, max_depth_tmp.shape).float()
                output = F.grid_sample(max_depth_tmp, grid, mode='nearest').reshape(B, N, C, H, W)  #平移后图像
                output = max_depth - output
                output_mask = ((output == max_depth) == False)
                output = output * output_mask
                output_list.append(output)
            grad = torch.cat(output_list, dim=2)  # [2, 6, 4, 256, 704]
            max_grad = torch.abs(grad).max(dim=2)[0].unsqueeze(2)
            img_metas[0].update({'max_grad': max_grad})
            #depth_ = depth
            depth = torch.cat([depth, grad], dim=2)  # [1, 6, 5, 448, 800]

            img_bev_feat, depth_dist, img_metas = self.lift_splat_shot_vis(img_feats_view, rots, trans, lidar2img_rt=lidar2img_rt, img_metas=img_metas
                                                                ,depth_lidar=depth)
            #print(img_bev_feat.size())
            if pts_feats is None:
                pts_feats = [img_bev_feat] ####cam stream only
            else:
                if self.lc_fusion:
                    if img_bev_feat.shape[2:] != pts_feats[0].shape[2:]:
                        img_bev_feat = F.interpolate(img_bev_feat, pts_feats[0].shape[2:], mode='bilinear', align_corners=True)
                    pts_feats = [self.reduc_conv(torch.cat([img_bev_feat, pts_feats[0]], dim=1))]
                    if self.se:
                        pts_feats = [self.seblock(pts_feats[0])]
        return dict(
            img_feats = img_feats,
            pts_feats = pts_feats,
            img_metas = img_metas,
            depth_dist = depth_dist
        )

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        img_metas = feature_dict['img_metas']
        depth_dist = feature_dict['depth_dist']
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
            if img_depth is not None:
                loss_depth = self.depth_dist_loss(depth_dist, img_depth, loss_method=self.img_depth_loss_method, img=img) * self.img_depth_loss_weight
                losses.update(img_depth_loss=loss_depth)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas, gt_bboxes_3d)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, img_metas]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, radar, img=None, rescale=False):
        """Test function without augmentaiton."""
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list
