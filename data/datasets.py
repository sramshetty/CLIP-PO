from collections import defaultdict
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import pandas as pd

import torch.nn as nn
from torch.utils.data import Dataset


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
        self.tokenizer.context_length = 2079
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