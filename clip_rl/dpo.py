import copy
from tqdm import tqdm
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F

import open_clip

from utils import patch_model


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
        reference_free: bool = False,
        peft_method: str = None,
        alpha: float = None
    ):
        super().__init__()

        self.device = device

        self.clip, self.train_preprocess, self.preprocess = open_clip.create_model_and_transforms(
            clip,
            pretrained=pretrained,
        )
        if peft_method == "a-clip":
            # freezing all weights but the attention in-projections: https://arxiv.org/pdf/2402.09613v1.pdf
            for p in self.clip.parameters():
                p.requires_grad = False
            for name, p in self.clip.named_parameters():
                if "attn.in_proj" in name:
                    p.requires_grad = True
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
        assert self.loss_type in {"ipo", "dpo"}

        self.label_smoothing = label_smoothing

        self.reference_free = reference_free or (self.label_smoothing == 0 and self.loss_type == "dpo")

        self.alpha = alpha
    
    @staticmethod
    def compute_reward(clip_model, images, tokens):
        image_feats = clip_model.encode_image(images, normalize=True)
        text_feats = clip_model.encode_text(tokens, normalize=True)
        
        text_logits = text_feats @ image_feats.T
        # image_logits = text_logits.T
        pref_logits = torch.diagonal(text_logits)
        rej_logits = torch.diagonal(text_logits[text_logits.size(0) // 2:])
        logits = torch.cat((pref_logits.unsqueeze(dim=0), rej_logits.unsqueeze(dim=0)), dim=0)
        probs = F.softmax(logits, dim=0)
        
        return probs[0], probs[1]

        # similarities = text_features @ image_feats.T
        # probs = (similarities + 1) / 2
        # pref_probs = torch.diagonal(probs)
        # rej_probs = torch.diagonal(probs[probs.size(0) // 2:])
        
        # return pref_probs, rej_probs
    
    def _compute_loss(
        self,
        policy_preferred_logps,
        policy_rejected_logps,
        reference_preferred_logps,
        reference_rejected_logps,
    ):
        policy_logratios = policy_preferred_logps / policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=policy_logratios.dtype, device=policy_logratios.device)
        else:
            ref_logratios = reference_preferred_logps / reference_rejected_logps
        logits = torch.log(policy_logratios / ref_logratios)

        if self.loss_type == "ipo":
            losses = (logits - 1/(2 * self.beta)) ** 2
        else:
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        preferred_rewards = (
            self.beta
            * (
                policy_preferred_logps.to(self.device)
                - reference_preferred_logps.to(self.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.device)
                - reference_rejected_logps.to(self.device)
            ).detach()
        )

        return losses.mean(), preferred_rewards, rejected_rewards

    def save(self, save_path):
        if self.alpha:
            # should model be updated in place for checkpoint saves?
            patch_model(self.frozen_clip, self.clip, self.alpha, save_path)
        else:
            torch.save(self.clip.state_dict(), save_path)
    
    def step(self, images, tokens):
        images = images.to(self.device)
        tokens = tokens.to(self.device)

        # compute frozen model's probs
        if not self.reference_free:
            with torch.no_grad():
                reference_preferred_logps, reference_rejected_logps = self.compute_reward(
                    self.frozen_clip, 
                    images,
                    tokens
                )
        
        # compute update model's probs
        policy_preferred_logps, policy_rejected_logps = self.compute_reward(
            self.clip,
            images,
            tokens
        )

        return self._compute_loss(
            policy_preferred_logps=policy_preferred_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_preferred_logps=reference_preferred_logps,
            reference_rejected_logps=reference_rejected_logps
        )
    
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

                loss, preferred_rewards, rejected_rewards = self.step(
                    images=batch['image'],
                    tokens=torch.cat([batch['preferred'], batch['rejected']]).squeeze()
                )
                loss.backward()

                optimizer.step()
                scheduler.step()

                reward_acc = (preferred_rewards > rejected_rewards).float()

                if i > 0 and i % 100 == 0:
                    print(f"Loss {i}/{len(dataloader)} = {loss}")
                    print(f"Reward Accuracy {i}/{len(dataloader)} = {reward_acc}")
            
            if (ep + 1) % save_freq == 0:
                self.save(save_path=self.save_path)
        
        self.save(save_path=self.save_path)


class CoCaDPOTrainer(nn.Module):
    def __init__(
        self,
        coca: str,
        pretrained: str,
        beta: int,
        save_path: str,
        loss_type: str,
        label_smoothing: float,
        lock_vision: bool,
        device: Union[str, torch.device],
        reference_free: bool = False,
        peft_method: str = None,
        alpha: float = None
    ):
        super().__init__()

        self.device = device

        self.coca, self.train_preprocess, self.preprocess = open_clip.create_model_and_transforms(
            coca,
            pretrained=pretrained,
        )
        if peft_method == "a-clip":
            # freezing all weights but the attention in-projections: https://arxiv.org/pdf/2402.09613v1.pdf
            for p in self.coca.parameters():
                p.requires_grad = False
            for name, p in self.coca.named_parameters():
                if "attn.in_proj" in name:
                    p.requires_grad = True
        if lock_vision:
            self.coca.visual.lock()
        self.coca.to(self.device)

        self.frozen_coca = copy.deepcopy(self.coca)
        for p in self.frozen_coca.parameters():
            p.requires_grad = False
        self.frozen_coca.eval()
        self.frozen_coca.to(self.device)
        
        self.tokenizer = open_clip.get_tokenizer(coca)

        self.beta = beta

        self.save_path = save_path

        self.loss_type = loss_type
        assert self.loss_type in {"ipo", "dpo"}

        self.label_smoothing = label_smoothing

        self.reference_free = reference_free or (self.label_smoothing == 0 and self.loss_type == "dpo")

        self.alpha = alpha
    
    @staticmethod
    def compute_reward(
        coca_model,
        images,
        tokens,
        average_log_prob: bool = False,
        label_pad_token_id: int = 0,
    ):
        outputs = coca_model(images, tokens)

        text_logits = outputs['text_features'] @ outputs['image_features'].chunk(2)[0].T
        # image_logits = outputs['text_features'].T
        pref_logits = torch.diagonal(text_logits)
        rej_logits = torch.diagonal(text_logits[text_logits.size(0) // 2:])
        mm_logits = torch.cat((pref_logits.unsqueeze(dim=0), rej_logits.unsqueeze(dim=0)), dim=0)
        probs = F.softmax(mm_logits, dim=0)

        # similarities = text_features @ image_feats.T
        # probs = (similarities + 1) / 2
        # pref_probs = torch.diagonal(probs)
        # rej_probs = torch.diagonal(probs[probs.size(0) // 2:])

        gen_logits = outputs['logits']
        labels = outputs['labels']

        # adapted from https://github.com/huggingface/trl/
        if gen_logits.shape[:-1] != labels.shape:
            raise ValueError("gen_Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        gen_logits = gen_logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(gen_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            gen_probs = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            gen_probs = (per_token_logps * loss_mask).sum(-1)
        
        pref_probs = (0.5 * probs[0]) + (0.5 * gen_probs[:gen_logits.size(0) // 2])
        rej_probs = (0.5 * probs[1]) + (0.5 * gen_probs[gen_logits.size(0) // 2:])
        
        return pref_probs, rej_probs
    
    def _compute_loss(
        self,
        policy_preferred_logps,
        policy_rejected_logps,
        reference_preferred_logps,
        reference_rejected_logps,
    ):
        # policy_logratios = policy_preferred_logps / policy_rejected_logps
        policy_logratios = policy_preferred_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=policy_logratios.dtype, device=policy_logratios.device)
        else:
            # ref_logratios = reference_preferred_logps / reference_rejected_logps
            ref_logratios = reference_preferred_logps - reference_rejected_logps
        # logits = torch.log(policy_logratios / ref_logratios)
        logits = policy_logratios - ref_logratios

        if self.loss_type == "ipo":
            losses = (logits - 1/(2 * self.beta)) ** 2
        else:
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        preferred_rewards = (
            self.beta
            * (
                policy_preferred_logps.to(self.device)
                - reference_preferred_logps.to(self.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.device)
                - reference_rejected_logps.to(self.device)
            ).detach()
        )

        return losses.mean(), preferred_rewards, rejected_rewards

    def save(self, save_path):
        if self.alpha:
            # should model be updated in place for checkpoint saves?
            patch_model(self.frozen_coca, self.coca, self.alpha, save_path)
        else:
            torch.save(self.coca.state_dict(), save_path)
    
    def step(self, images, tokens):
        images = images.to(self.device)
        tokens = tokens.to(self.device)

        # compute frozen model's probs
        if not self.reference_free:
            with torch.no_grad():
                reference_preferred_logps, reference_rejected_logps = self.compute_reward(
                    self.frozen_coca,
                    images,
                    tokens,
                    average_log_prob=self.loss_type=="ipo"
                )
        
        # compute update model's probs
        policy_preferred_logps, policy_rejected_logps = self.compute_reward(
            self.coca,
            images,
            tokens,
            average_log_prob=self.loss_type=="ipo"
        )

        return self._compute_loss(
            policy_preferred_logps=policy_preferred_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_preferred_logps=reference_preferred_logps,
            reference_rejected_logps=reference_rejected_logps
        )
    
    def train(
        self,
        optimizer,
        scheduler,
        dataloader,
        num_epochs = 5,
        save_freq = 1,
    ):
        self.coca.train()

        for ep in range(num_epochs):
            for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {ep}")):

                optimizer.zero_grad()

                loss, preferred_rewards, rejected_rewards = self.step(
                    images=batch['image'].repeat(2, 1, 1, 1),
                    tokens=torch.cat([batch['preferred'], batch['rejected']]).squeeze()
                )
                loss.backward()

                optimizer.step()
                scheduler.step()

                reward_acc = (preferred_rewards > rejected_rewards).float()

                if i > 0 and i % 100 == 0:
                    print(f"Loss {i}/{len(dataloader)} = {loss}")
                    print(f"Reward Accuracy {i}/{len(dataloader)} = {reward_acc}")
            
            if (ep + 1) % save_freq == 0:
                self.save(save_path=self.save_path)
        
        self.save(save_path=self.save_path)
