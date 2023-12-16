from argparse import ArgumentParser
import json
import os
from PIL import Image
import random
from typing import List, Union 
from tqdm import tqdm

import numpy as np
import pandas as pd

from pycocotools.coco import COCO

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as transforms

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForVision2Seq,
    LlamaTokenizer
)


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
    parser.add_argument(
        "--retrieve_samples",
        action="store_true",
        help="whether to retrieve images"
    )
    parser.add_argument(
        "--add_spatial",
        action="store_true",
        help="whether to generate captions with spatial information for retrieved images"
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
            # "vflip": transforms.functional.vertical_flip,
            "color_jitter": transforms.ColorJitter(brightness=0.5, hue=0.5)
        }
        transform_type = random.choice(list(transform_map.keys()))
        aug_img = transform_map[transform_type](img)

        if not os.path.exists("./data/augments"):
            os.makedirs("./data/augments")

        aug_path = os.path.abspath(
            os.path.join(
                "./data/augments",
                img_path.split("/")[-1].split(".")[0] + "_" + transform_type + ".png"
            )
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


def add_spatial(
    model: nn.Module,
    tokenizer: nn.Module,
    parquet_path: str,
    caption_type: str = "detailed",
    batch_size: int = 1
):
    data_df = pd.read_parquet(parquet_path)

    detailed_captions = []
    with open("temp_captions.json") as f:
        try:
            detailed_captions = json.load(f)
        except:
            pass
    
    rows_to_skip = len(detailed_captions)
    print(f"Skipping {rows_to_skip} rows")

    query = 'Describe this image in detail with the spatial layout and colors of each object.'
    for row_idx, row in tqdm(enumerate(data_df.itertuples()), total=len(data_df)):
        
        if row_idx < rows_to_skip:
            continue
        
        images = []
        for img_path in row.image_paths:
            images.append(Image.open(img_path).convert('RGB'))
        
        row_captions = []
        for i in range(0, len(images), batch_size):
            inputs = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=[],
                images=images[i:i+batch_size]
            )
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
                'images': [[inputs['images'][0].to(device).to(torch.float16)]],
            }
            gen_kwargs = {"max_length": 2048, "do_sample": False}

            
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                row_captions.append(tokenizer.decode(outputs[0]))
        
        detailed_captions.append(row_captions)
            
        if row_idx > 0 and row_idx % 100 == 0:
            with open("temp_captions.json", "w") as f:
                json.dump(detailed_captions, f)

    with open("temp_captions.json", "w") as f:
        json.dump(detailed_captions, f)

    data_df['detailed_gen_captions'] = detailed_captions
    data_df.to_parquet(parquet_path)

    os.remove("temp_captions.json")


if __name__ == "__main__":
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()
    caption_file = args.caption_file_path
    img_dir = os.path.abspath(args.img_dir)

    coco_captions = COCO(caption_file)
    captions = [cap_dict["caption"] for cap_dict in coco_captions.anns.values()]

    image_files = [os.path.join(img_dir, file) for file in sorted(os.listdir(img_dir))]

    text_embeds = torch.Tensor(np.load(args.text_embs_path))
    image_embeds = torch.Tensor(np.load(args.img_embs_path))

    if args.retrieve_samples:
        generate_retrieval_data(
            captions=captions,
            image_files=image_files,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            output_parquet=args.output_parquet,
            num_samples=args.num_samples,
        )

    if args.add_spatial:
        print("Adding spatial captions")
        # model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
        # processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

        tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm-chat-hf",
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            trust_remote_code=True,
        ).eval()

        add_spatial(model, tokenizer, args.output_parquet)
