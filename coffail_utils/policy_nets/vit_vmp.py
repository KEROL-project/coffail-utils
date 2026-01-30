import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

class ViTVMPNet(nn.Module):
    def __init__(self, number_of_actions: int,
                 vit_model_name: str='google/vit-base-patch16-224',
                 pretrained_model_path: str=None):
        super(ViTVMPNet, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.image_processor = AutoImageProcessor.from_pretrained(vit_model_name, use_fast=True)
        self.feature_extraction_model = AutoModel.from_pretrained(vit_model_name)

        self.action_extractor = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, number_of_actions),
            nn.Tanh()
        )

        if pretrained_model_path:
            print('Loading saved model {0}'.format(pretrained_model_path))
            self.load_state_dict(torch.load(pretrained_model_path))

        self.to(self.device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        processed_image = self.image_processor(image, return_tensors="pt").to(self.device)
        image_features = self.feature_extraction_model(**processed_image).pooler_output.squeeze().detach()
        actions = self.action_extractor(image_features).cpu()
        return actions

    def save(self, model_path):
        """Saves the model to the given path.

        Keyword arguments:
        model_path: str -- file where the model should be saved

        """
        torch.save(self.state_dict(), model_path)
