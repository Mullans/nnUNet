import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDAOrd0 import nnUNetTrainer_DASegOrd0
from nnunetv2.training.loss.compound_losses import Tversky_and_CE_loss, DistanceBCELoss, DistanceComboLoss
import numpy as np


class nnUNetTrainerDistanceComboLoss(nnUNetTrainer):
    tversky_kwargs = {
        'smooth': 0.0,
        'alpha': 0.3,
        'beta': 0.7,
        'gamma': 1.333
    }
    ce_kwargs = {
        'label_smoothing': 1e-4
    }
    loss_weights = [1, 1, 1]
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"
        # alpha = beta = 0.5 -> DiceLoss
        # alpha = beta = 1 -> Tanimoto coefficient
        # alpha + beta = 1 -> F_beta scores

        loss = DistanceComboLoss(
            self.tversky_kwargs,
            self.ce_kwargs,
            weight_ce=self.loss_weights[0],
            weight_tversky=self.loss_weights[1],
            weight_dist=self.loss_weights[2],
            ignore_label=self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100,
            dist_err_type="both"
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