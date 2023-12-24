from typing import Union

import torch
import torch.nn as nn

import open_clip


class RewardCLIP(nn.Module):
    def __init__(
        self,
        clip: str,
        pretrained: str,
        freeze_vision: bool,
        alpha: int,
        device: Union[str, torch.device],
    ):
        super().__init__()

        self.device = device

        self.clip, self.train_preprocess, self.preprocess = open_clip.create_model_and_transforms(
            clip,
            pretrained=pretrained,
        )
        self.clip.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(clip)

        if freeze_vision:
            self.clip.lock_image_tower()
        
        # if use_lora:
        #     text_encoder = self.clip.transformer

        self.alpha = alpha  # factor to multiply sim difference; may want to make this learnable
    
    def load(self, state_dict):
        self.clip.load_state_dict(state_dict)
    
    def encode_texts(self, tokens):
        text_features = self.clip.encode_text(tokens.to(self.device))
        text_features = text_features / text_features.clone().norm(dim=-1, keepdim=True)
        return text_features

    def encode_images(self, images):
        image_feats = self.clip.encode_image(images.to(self.device))
        image_feats = image_feats / image_feats.clone().norm(dim=-1, keepdim=True)
        return image_feats
    
    def forward(
        self,
        input_embs: torch.Tensor,
        neg_input_embs: torch.Tensor,
        image_embs: torch.Tensor,
        return_sims: bool = False,
    ):
        image_embs = image_embs.T
        rewards_pos = (input_embs @ image_embs).diagonal()
        rewards_neg = (neg_input_embs @ image_embs).diagonal()
        loss = -nn.functional.logsigmoid((rewards_pos - rewards_neg) * self.alpha).mean()

        if return_sims:
            return loss, {"pos_sim": rewards_pos, "neg_sim": rewards_neg}
        
        return loss
