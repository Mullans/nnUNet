import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDAOrd0 import nnUNetTrainer_DASegOrd0
from nnunetv2.training.loss.compound_losses import Tversky_and_CE_loss
import numpy as np


class nnUNetTrainerTverskyCELoss(nnUNetTrainer):
    tversky_kwargs = {
        'smooth': 0.0,
        'alpha': 0.3, # Weight constant that penalize model for FPs (False Positives)
        'beta': 0.7, # Weight constant that penalize model for FNs (False Negatives)
        'gamma': 1.333 #  Constant that squares the error function. Defaults to ``1.0``
    }
    ce_kwargs = {
        'label_smoothing': 1e-4
    }
    loss_weights = [1, 1]
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        # alpha = beta = 0.5 -> DiceLoss
        # alpha = beta = 1 -> Tanimoto coefficient
        # alpha + beta = 1 -> F_beta scores

        loss = Tversky_and_CE_loss(
            self.tversky_kwargs,
            self.ce_kwargs,
            weight_ce=self.loss_weights[0],
            weight_tversky=self.loss_weights[1],
            ignore_label=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100
        )

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerTverskyCELoss_V2(nnUNetTrainerTverskyCELoss):
    tversky_kwargs = {
        'smooth': 1e-5,
        'alpha': 0.3,
        'beta': 0.7,
        'gamma': 1.0
    }
    ce_kwargs = {
        'label_smoothing': 1e-5
    }
    loss_weights = [1, 2]


class nnUNetTrainerTverskyCELoss_FNFocus(nnUNetTrainerTverskyCELoss):
    tversky_kwargs = {
        'smooth': 1e-5,
        'alpha': 0.7,
        'beta': 0.3,
        'gamma': 1.333
    }
    ce_kwargs = {
        'label_smoothing': 1e-5
    }
    loss_weights = [1, 2]


class nnUNetTrainerTverskyCELoss_FNFocus2(nnUNetTrainerTverskyCELoss):
    tversky_kwargs = {
        'smooth': 1e-5,
        'alpha': 0.7,
        'beta': 0.3,
        'gamma': 1.333
    }
    ce_kwargs = {
        'label_smoothing': 1e-5
    }
    loss_weights = [1, 3]


class nnUNetTrainerTverskyCELoss_FNFocus3(nnUNetTrainerTverskyCELoss):
    tversky_kwargs = {
        'smooth': 1e-5,
        'alpha': 0.7,
        'beta': 0.3,
        'gamma': 2.0
    }
    ce_kwargs = {
        'label_smoothing': 1e-5
    }
    loss_weights = [1, 3]


class nnUNetTrainerTverskyCELoss_DAOrd0(nnUNetTrainer_DASegOrd0):
    tversky_kwargs = {
        'smooth': 1e-5,
        'alpha': 0.7,
        'beta': 0.3,
        'gamma': 2.0
    }
    ce_kwargs = {
        'label_smoothing': 1e-5
    }
    loss_weights = [1, 3]
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        # alpha = beta = 0.5 -> DiceLoss
        # alpha = beta = 1 -> Tanimoto coefficient
        # alpha + beta = 1 -> F_beta scores

        loss = Tversky_and_CE_loss(
            self.tversky_kwargs,
            self.ce_kwargs,
            weight_ce=self.loss_weights[0],
            weight_tversky=self.loss_weights[1],
            ignore_label=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100
        )

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss