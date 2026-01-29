import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim

from coffail_utils.policy_nets.vit_vmp import ViTVMPNet
from coffail_utils.policies.base_policy import BasePolicy

class ViTVMP(BasePolicy):
    def __init__(self, number_of_actions: int=6,
                 pretrained_model_path: str=None):
        super(ViTVMP, self).__init__()
        self.policy_net = ViTVMPNet(number_of_actions=number_of_actions,
                                    pretrained_model_path=pretrained_model_path)

    def act(self, image: torch.Tensor) -> torch.Tensor:
        action = self.policy_net.forward(image=image)
        return action

    def save(self, path: str) -> None:
        """Saves the policy to the given path.

        Keyword arguments:
        path: str -- file where the policy should be saved

        """
        self.policy_net.save(path)
