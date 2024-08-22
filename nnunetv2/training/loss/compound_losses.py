import segmentation_models_pytorch
import torch
from torch import nn

from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result



# ## CUSTOM LOSSES ## #
class Tversky_and_CE_loss(nn.Module):
    def __init__(self, tversky_kwargs, ce_kwargs, weight_ce=1, weight_tversky=1, ignore_label=None,
                 tversky_class=segmentation_models_pytorch.losses.TverskyLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param tverksy_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_tversky:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_tv = weight_tversky
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.tv = tversky_class(mode="multiclass", from_logits=True, ignore_index=0, **tversky_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_tv = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_tv = target
            mask = None

        tv_loss = self.tv(net_output, target_tv) \
            if self.weight_tv != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_tv * tv_loss
        return result


import numpy as np
import scipy.ndimage as nd
class DistanceBCELoss(nn.Module):
    def __init__(self, under_weight: float = 1.0, over_weight: float = 1.0):
        """_summary_
        Add a distance transform to the binary cross entropy loss to penalize errors based on their distance to the target.
        :param err_type: str, either 'over', 'under', or 'both', whether to penalize over- or under-segmentation errors or both
        """
        super().__init__()
        self.under_weight = under_weight
        self.over_weight = over_weight

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        target_label = torch.zeros_like(net_output, dtype=net_output.dtype).scatter_(1, target.long(), True)
        target_mask = (target > 0.5).float().detach().cpu().numpy()
        target_dist = np.zeros_like(target_mask)
        if self.under_weight != 0.0:
            target_dist += np.stack([
                nd.distance_transform_edt(target_mask[idx]) for idx in range(target_mask.shape[0])
            ])
        elif self.over_weight != 0.0:
            target_mask = 1 - target_mask
            target_dist += np.stack([
                nd.distance_transform_edt(target_mask[idx]) for idx in range(target_mask.shape[0])
            ])
        target_dist += 1
        target_dist = torch.from_numpy(target_dist).to(device=target.device, dtype=target.dtype)
        result = torch.nn.functional.binary_cross_entropy_with_logits(net_output, target_label, weight=target_dist)
        return result

class DistanceComboLoss(nn.Module):
    def __init__(self, tversky_kwargs, ce_kwargs, dist_kwargs, weight_ce=1, weight_tversky=1, weight_dist=1,
                 ignore_label=None, tversky_class=segmentation_models_pytorch.losses.TverskyLoss):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_tv = weight_tversky
        self.weight_ce = weight_ce
        self.weight_dist = weight_dist
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.tv = tversky_class(mode="multiclass", from_logits=True, ignore_index=0, **tversky_kwargs)
        self.de = DistanceBCELoss(**dist_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_tv = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_tv = target
            mask = None

        tv_loss = self.tv(net_output, target_tv) \
            if self.weight_tv != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        dist_loss = self.de(net_output, target_tv) if self.weight_dist != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_tv * tv_loss + self.weight_dist * dist_loss
        return result