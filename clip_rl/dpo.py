import copy
from tqdm import tqdm
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F

import open_clip


class ClipDPOTrainer(nn.Module):
    def __init__(
        self,
        clip: str,
        pretrained: str,
        beta: int,
        save_path: str,
        loss_type: str,
        label_smoothing: float,
        lock_vision: bool,
        device: Union[str, torch.device],
    ):
        super().__init__()

        self.device = device

        self.clip, self.train_preprocess, self.preprocess = open_clip.create_model_and_transforms(
            clip,
            pretrained=pretrained,
        )
        if lock_vision:
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

        self.loss_type = loss_type
        self.label_smoothing = label_smoothing

        assert self.loss_type in {"ipo", "dpo"}
    
    def _compute_reward(self, clip_model, images, tokens):
        image_feats = clip_model.encode_image(images.to(self.device))
        image_feats = image_feats / image_feats.clone().norm(dim=-1, keepdim=True)
        text_features = clip_model.encode_text(tokens.to(self.device))
        text_features = text_features / text_features.clone().norm(dim=-1, keepdim=True)
        
        similarities = text_features @ image_feats.T
        probs = (similarities + 1) / 2
        pref_probs = torch.diagonal(probs)
        rej_probs = torch.diagonal(probs[probs.size(0) // 2:])
        rewards = pref_probs - rej_probs
        
        return rewards
    
    def _compute_loss(self, frozen_rewards, update_rewards):
        logits = update_rewards - frozen_rewards
        if self.loss_type == "ipo":
            loss = (logits - 1/(2 * self.beta)) ** 2
        else:
            loss = -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) - F.logsigmoid(-self.beta * logits) * self.label_smoothing

        return loss.mean()

    def save(self, save_path):
        torch.save(self.clip.state_dict(), save_path)
    
    def step(self, images, tokens):
        # compute frozen model's probs
        with torch.no_grad():
            frozen_rewards = self._compute_reward(self.frozen_clip, images, tokens)
        
        # compute update model's probs
        update_rewards = self._compute_reward(self.clip, images, tokens)

        return self._compute_loss(frozen_rewards=frozen_rewards, update_rewards=update_rewards), frozen_rewards.detach(), update_rewards.detach()
    
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

                loss, frozen_rewards, update_rewards = self.step(
                    images=batch['image'],
                    tokens=torch.cat([batch['preferred'], batch['rejected']]).squeeze()
                )
                loss.backward()

                optimizer.step()
                scheduler.step()

                frozen_acc = (frozen_rewards > 1).sum() / dataloader.batch_size
                update_acc = (update_rewards > 1).sum() / dataloader.batch_size

                if i > 0 and i % 100 == 0:
                    print(f"Loss {i}/{len(dataloader)} = {loss}")
                    print(f"Frozen Rewards {i}/{len(dataloader)} = {frozen_rewards}")
                    print(f"Update Rewards {i}/{len(dataloader)} = {update_rewards}")
                    print(f"Frozen Reward Accuracy {i}/{len(dataloader)} = {frozen_acc}")
                    print(f"Update Reward Accuracy {i}/{len(dataloader)} = {update_acc}")
            
            if (ep + 1) % save_freq == 0:
                self.save(save_path=self.save_path)
        
        self.save(save_path=self.save_path)