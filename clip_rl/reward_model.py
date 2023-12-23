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

        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
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
    
    def encode_texts(self, texts):
        texts = self.tokenizer(texts)
        text_features = self.clip.encode_text(texts.to(self.device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_images(self, images):
        image_feats = self.clip.encode_image(torch.cat(images, dim=0).to(self.device))
        image_feats /= image_feats.norm(dim=-1, keepdim=True)
        return image_feats
    
    def forward(
        self,
        input_ids: torch.Tensor,
        neg_input_ids: torch.Tensor,
        images: torch.Tensor,
        return_sims: bool = False,
    ):
        # currently only handles binary case
        text_feats = self.encode_texts(input_ids)
        neg_text_feats = self.encode_texts(neg_input_ids)
        image_feats = self.encode_images(images)

        rewards_pos = text_feats @ image_feats.T
        rewards_neg = neg_text_feats @ image_feats.T
        loss = -nn.functional.logsigmoid((rewards_pos - rewards_neg) * self.alpha).mean()

        if return_sims:
            return loss, {"pos_sim": rewards_pos, "neg_sim": rewards_neg}
        
        return loss
