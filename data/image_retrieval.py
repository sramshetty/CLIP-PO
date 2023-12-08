from argparse import ArgumentParser
import os
import random
from typing import List, Union 
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms.v2 as transforms

from pycocotools.coco import COCO


def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument(
        "--caption_file_path",
        type=str,
        help="path to coco caption file"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        help="path to coco image directory"
    )
    parser.add_argument(
        "--text_embs_path",
        type=str,
        help="path to numpy file containing text embeddings"
    )
    parser.add_argument(
        "--img_embs_path",
        type=str,
        help="path to numpy file containing image embeddings"
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        help="path for output parquet"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="number of samples to generate"
    )

    return parser.parse_args()


def create_augments(image_files: List[str], num_augments: int = 4):
    """Create augments of retrieved images"""
    aug_paths = []
    while num_augments > 0:
        img_path =  random.choice(image_files)
        img = torchvision.io.read_image(img_path)
        transform_map = {
            "hflip": transforms.functional.horizontal_flip,
            "vflip": transforms.functional.vertical_flip,
            "color_jitter": transforms.ColorJitter(brightness=0.5, hue=0.5)
        }
        transform_type = random.choice(list(transform_map.keys()))
        aug_img = transform_map[transform_type](img)

        if not os.path.exists("./data/augments"):
            os.makedirs("./data/augments")

        aug_path = os.path.join(
            "./data/augments",
            img_path.split("/")[-1].split(".")[0] + "_" + transform_type + ".png"
        )
        if aug_path in aug_paths:
            continue

        torchvision.io.write_png(aug_img, aug_path)
        aug_paths.append(aug_path)
        num_augments -= 1

    return aug_paths


def generate_retrieval_data(
    captions: List[str],
    image_files: List[str],
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    output_parquet: str,
    num_samples: int = 1000,
):
    """Create a dataset of retrieved images for N random captions"""

    image_retrieval_samples = []
    for _ in tqdm(range(num_samples)):
        text_id = random.randint(0, len(text_embeds))
        text = captions[text_id]

        topk = torch.topk(text_embeds[text_id].unsqueeze(0) @ image_embeds.T, k=5)
        sims = topk.values[0].numpy().tolist()
        max_sim = max(sims)
        topk_indices = topk.indices[0].numpy().tolist()

        img_paths = [image_files[image_id] for image_id, sim in zip(topk_indices, sims) if sim >= max_sim - 0.05]
        if len(img_paths) < 2:
            img_paths.append(image_files[topk_indices[1]])
        
        # guarantees at least 4 images and at most 9 images
        num_augments = min(random.randint(4, 9) - len(img_paths), len(img_paths))
        aug_paths = create_augments(img_paths, num_augments=num_augments)

        sample = {"text_id": text_id, "text": text, "image_ids": topk_indices, "similarities": sims, "image_paths": img_paths + aug_paths}
        image_retrieval_samples.append(sample)

    ranking_df = pd.DataFrame(image_retrieval_samples)
    ranking_df.to_parquet(output_parquet)


if __name__ == "__main__":
    args = parse_args()
    caption_file = args.caption_file_path
    img_dir = args.img_dir

    coco_captions = COCO(caption_file)
    captions = [cap_dict["caption"] for cap_dict in coco_captions.anns.values()]

    image_files = [os.path.join(img_dir, file) for file in sorted(os.listdir(img_dir))]

    text_embeds = torch.Tensor(np.load(args.text_embs_path))
    image_embeds = torch.Tensor(np.load(args.img_embs_path))

    generate_retrieval_data(
        captions=captions,
        image_files=image_files,
        text_embeds=text_embeds,
        image_embeds=image_embeds,
        output_parquet=args.output_parquet,
        num_samples=args.num_samples,
    )