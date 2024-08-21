from typing import List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss, Tversky_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
import segmentation_models_pytorch as smp

from .medformer_model import MedFormer

class nnUNetTrainerMedform(nnUNetTrainerDA5):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 1000, verbose=False):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device, num_epochs=num_epochs, verbose=verbose)
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        model = MedFormer(num_input_channels, num_output_channels)
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        if hasattr(mod, 'decoder'):
            if hasattr(mod.decoder, 'deep_supervision'):
                mod.decoder.deep_supervision = enabled


class nnUNetTrainerMedform_Loss2(nnUNetTrainerMedform):
    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': True, 'ddp': self.is_ddp}, {}, weight_ce=2, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerMedform_Tversky(nnUNetTrainerMedform):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise NotImplementedError("Tversky loss does not support regions")
            # loss = DC_and_BCE_loss({},
            #                        {'batch_dice': self.configuration_manager.batch_dice,
            #                         'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
            #                        use_ignore_label=self.label_manager.ignore_label is not None,
            #                        dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = Tversky_and_CE_loss({'smooth': 1e-5, 'alpha': 0.3, 'beta': 0.7, 'gamma': 1.33},
                                       {},
                                       weight_ce=1,
                                       weight_tversky=1,
                                       ignore_label=self.label_manager.ignore_label)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerX(nnUNetTrainerMedform_Tversky):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        model = smp.Unet('xception',
                         in_channels=num_input_channels,
                         classes=num_output_channels,
                         encoder_depth=5,
                         encoder_weights=None,
                         decoder_attention_type='scse')
        return model


class nnUNetTrainerMA(nnUNetTrainerMedform):
    # NOTE - uses default nnUNet loss - equal Dice and CE
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        model = smp.MAnet('xception',
                          in_channels=num_input_channels,
                          classes=num_output_channels,
                          encoder_depth=5,
                          encoder_weights=None)
        return model