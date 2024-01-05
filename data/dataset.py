from collections import defaultdict
from glob import glob
from imageio import imread
import os
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pycocotools.coco import COCO


class RetrievalDataset(Dataset):
    """Dataset containing samples of (
        Image Embedding,
        Positive Text Embedding,
        Negative Text Embedding
    )

    Creates these binary text samples from K ranked text retrievals.
    """
    def __init__(
        self,
        model: nn.Module,
        parquet: str,
        split: str = "test",
        seed: int = 42
    ):
        self.model = model
        self.tokenizer = self.model.tokenizer
        # self.tokenizer.context_length = 308
        self.preprocess = self.model.train_preprocess

        self.df = pd.read_parquet(parquet)
        self.num_samples = len(self.df)
        train_df = self.df.sample(n=int(self.num_samples * 0.8), random_state=seed)
        if split == "test":
            self.df = self.df.drop(train_df.index)
            self.preprocess = self.model.preprocess
        else:
            self.df = train_df
        
        self.data_list = []
        self.create_samples()
    
    def create_samples(self):
        for row in tqdm(self.df.itertuples(), total=len(self.df), desc="creating samples"):
            image_paths = row.image_paths
            captions = row.detailed_gen_captions

            img_groups = self.group_sample_by_image(image_paths)
            for i, group in enumerate(img_groups):
                # if group has more than 1 sample, first index is base image
                # base image caption in our positive caption
                if len(group) > 1:
                    for idx in group[1:]:
                        self.data_list.append({
                            'caption': captions[group[0]],
                            'negative': captions[idx],
                            'image_path': image_paths[group[0]]
                        })

                for aux_group in img_groups[i+1:]:
                    for aux_idx in aux_group:
                        self.data_list.append({
                            'caption': captions[group[0]],
                            'negative': captions[aux_idx],
                            'image_path': image_paths[group[0]]
                        })
    
    def group_sample_by_image(self, image_paths):
        # Get indices with same base image (some samples are augment of base image)
        base_img_ids = defaultdict(list)
        for i, path in enumerate(image_paths):
            filename = Path(path).stem
            base_file = filename.split("_")[0]
            base_img_ids[base_file].append(i)
        
        # Create buckets of ids so we can create as many binary samples as posssible
        counts2_to_ids = [[] for _ in range(len(image_paths) + 1)]
        for indices in base_img_ids.values():
            counts2_to_ids[len(indices)].append(indices)
        
        # flatten buckets in order
        grouped_ids = []
        for bucket in counts2_to_ids[::-1]:
            if bucket:
                grouped_ids += bucket
        
        return grouped_ids

    def __getitem__(self, index):
        item = self.data_list[index]
        caption, negative, img_path = item['caption'], item['negative'], item['image_path']
        inputs = self.model.encode_texts(self.tokenizer(caption))
        negatives = self.model.encode_texts(self.tokenizer(negative))
        image = self.model.encode_images(self.preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0))
        res = {
            'input_emb': inputs.squeeze(),
            'negative_emb': negatives.squeeze(),
            'image_emb': image.squeeze(),
        }

        return res
    
    def __len__(self):
        return len(self.data_list)


class Cc12mApeDataset(Dataset):
    """Dataset containing samples of (
        Caption,
        CLIP Text Embedding,
    )

    Used for alignment between text encoder and another pretrained encoder.
    """

    def __init__(
        self,
        clip,
        preprocess,
        tokenizer,
        data_tsv,
        split: str = "test",
        seed: int = 42,
    ):
        self.clip = clip
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        self.df = pd.read_csv(data_tsv, sep="\t", names=["img_url", "caption"])
        self.num_samples = len(self.df)
        train_df = self.df.sample(n=int(self.num_samples * 0.8), random_state=seed)
        if split == "test":
            self.df = self.df.drop(train_df.index)
        else:
            self.df = train_df
        
        self.captions = self.df["caption"].to_list()
        self.image_urls = self.df["img_url"].to_list()
    
    def __getitem__(self, index, device="cuda"):
        while True:
            caption = self.captions[index]
            img_url = self.image_urls[index]

            text_emb = self.clip.encode_text(self.tokenizer(caption).to(device))
            text_emb = text_emb / text_emb.clone().norm(dim=-1, keepdim=True)
            try:
                img = imread(img_url)
            except:
                index = random.randint(0, self.__len__() - 1)
                continue
            image_emb = self.clip.encode_image(
                self.preprocess(Image.fromarray(img).convert('RGB')).unsqueeze(0).to(device)
            )
            image_emb = image_emb / image_emb.clone().norm(dim=-1, keepdim=True)
            res = {
                'caption': caption,
                'text_emb': text_emb.squeeze(),
                'image_emb': image_emb.squeeze()
            }
            break

        return res
    
    def __len__(self):
        return len(self.captions)
    

class CocoAlignDataset(Dataset):
    """Dataset containing samples of (
        Caption,
        CLIP Text Embedding,
    )

    Used for alignment between text encoder and another pretrained encoder.
    """

    def __init__(
        self,
        clip,
        preprocess,
        tokenizer,
        caption_file,
        image_dir,
        text_embeds_path = None,
        img_embeds_path = None,
        split: str = "test",
        seed: int = 42,
    ):
        self.clip = clip
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        self.image_dir = image_dir
        files = sorted(os.listdir(self.image_dir))
        # store img embedding index  for a given img id, since many captions share the same image
        self.files_dict = {int(f.split(".")[0]): i for i, f in enumerate(files)}

        coco_dict = COCO(caption_file).anns
        self.captions = [cap_dict["caption"] for cap_dict in coco_dict.values()]
        self.image_ids = [cap_dict["image_id"] for cap_dict in coco_dict.values()]
        self.text_embeds = torch.Tensor(np.load(text_embeds_path)) if text_embeds_path else None
        self.image_embeds = torch.Tensor(np.load(img_embeds_path)) if img_embeds_path else None

        random.seed(seed)
        self.num_samples = len(self.captions)
        self.shuffle_ids = list(range(self.num_samples))
        random.shuffle(self.shuffle_ids)

        train_idx = int(self.num_samples * 0.8)
        if split == "test":
            self.shuffle_ids = self.shuffle_ids[train_idx:]
        else:
            self.shuffle_ids = self.shuffle_ids[:train_idx]
    
    def __getitem__(self, index, device="cuda"):
        index = self.shuffle_ids[index]
        caption = self.captions[index]
        img_id = self.image_ids[index]
        
        if self.text_embeds is None:
            text_emb = self.clip.encode_text(self.tokenizer(caption).to(device))
            text_emb = text_emb / text_emb.clone().norm(dim=-1, keepdim=True)
            text_emb = text_emb.squeeze()
        else:
            text_emb = self.text_embeds[index].to(device)

        if self.image_embeds is None:
            img_path = glob(os.path.join(self.image_dir, str(img_id).zfill(12) + "*"))[-1]
            image_emb = self.clip.encode_image(
                self.preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
            )
            image_emb = image_emb / image_emb.clone().norm(dim=-1, keepdim=True)
            image_emb = image_emb.squeeze()
        else:
            img_emb_idx = self.files_dict[img_id]
            image_emb = self.image_embeds[img_emb_idx].to(device)

        res = {
            'caption': caption,
            'text_emb': text_emb,
            'image_emb': image_emb
        }

        return res
    
    def __len__(self):
        return len(self.shuffle_ids)