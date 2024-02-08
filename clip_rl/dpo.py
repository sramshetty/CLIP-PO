import copy
from tqdm import tqdm
from typing import Union

import torch
from torch import nn

import open_clip


class ClipDPOTrainer(nn.Module):
    def __init__(
        self,
        clip: str,
        pretrained: str,
        beta: int,
        save_path: str,
        device: Union[str, torch.device],
    ):
        super().__init__()

        self.device = device

        self.clip, self.train_preprocess, self.preprocess = open_clip.create_model_and_transforms(
            clip,
            pretrained=pretrained,
        )
        self.clip.lock_image_tower()
        self.clip.to(self.device)

        self.frozen_clip = copy.deepcopy(self.clip)
        for p in self.frozen_clip.parameters():
            p.requires_grad = False
        self.frozen_clip.eval()
        self.frozen_clip.to(self.device)
        
        self.tokenizer = open_clip.get_tokenizer(clip)

        self.beta = beta

        self.save_path = save_path
    
    def _compute_reward(self, clip_model, images, tokens):
        image_feats = clip_model.encode_image(images.to(self.device))
        image_feats = image_feats / image_feats.clone().norm(dim=-1, keepdim=True)
        text_features = clip_model.encode_text(tokens.to(self.device))
        text_features = text_features / text_features.clone().norm(dim=-1, keepdim=True)
        
        similarities = image_feats @ text_features.T
        probs = (similarities + 1) / 2
        scores = probs.split(probs.size(1) // 2, dim=1)
        reward = scores[0] / scores[1]

        return reward
    
    def _compute_loss(self, frozen_reward, update_reward):
        return -nn.functional.logsigmoid(self.beta * torch.log(update_reward / frozen_reward)).mean()

    def save(self, save_path):
        torch.save(self.clip.state_dict(), save_path)
    
    def step(self, images, tokens):
        # compute frozen model's probs
        frozen_reward = self._compute_reward(self.frozen_clip, images, tokens)
        
        # compute update model's probs
        update_reward = self._compute_reward(self.clip, images, tokens)

        return self._compute_loss(frozen_reward=frozen_reward, update_reward=update_reward)
    
    def train(
        self,
        optimizer,
        scheduler,
        dataloader,
        num_epochs = 5,
        save_freq = 1,
    ):
        self.clip.train()

        for ep in range(num_epochs):
            for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {ep}")):

                optimizer.zero_grad()

                loss = self.step(
                    images=batch['image'],
                    tokens=torch.cat([batch['preferred'], batch['rejected']]).squeeze()
                )
                loss.backward()

                optimizer.step()
                scheduler.step()

                if i > 0 and i % 100 == 0:
                    print(f"Loss {i}/{len(dataloader)} = {loss}")
            
            if (ep + 1) % save_freq == 0:
                self.save(save_path=self.save_path)
        
        self.save(save_path=self.save_path)