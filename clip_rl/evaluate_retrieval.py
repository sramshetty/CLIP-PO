from argparse import ArgumentParser
from glob import glob
import os
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch

from pycocotools.coco import COCO

import open_clip


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help="path to the image dir"
    )
    parser.add_argument(
        '--caption_file',
        type=str,
        required=True,
        help="path to coco caption file"
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        default="coco",
        help="type of dataset to inform image and caption loading"
    )

    parser.add_argument(
        '--model',
        type=str,
        default="ViT-H-14",
        help="open_clip model name"
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        default="laion2b_s32b_b79k",
        help="open_clip pretraining dataset associated with model"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help="batch size for encoding inputs"
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=1000,
        help="batch size for metric calculation"
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help="number of randomized iterations to average over"
    )

    args = parser.parse_args()

    assert os.path.exists(args.caption_file) if args.dataset_type == "coco" else True

    return args


def build_features(model, tokenizer, imgs, caps):
    image_input = torch.tensor(np.stack(imgs)).to(device)
    text_tokens = tokenizer(["This is " + desc for desc in caps]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return image_features, text_features


def evaluate(similarities):
    """
    Compute MRR and batch recall stats for a given similarity matrix.
    """
    mrr = 0
    tp = 0  # true positives
    fn = 0  # false negatives
    for cap in range(len(similarities)):
        # Since 1-1 image-text pair, retrieve similarity value of the actual pair
        correct_sim = similarities[cap][cap]
        # Get indices in descending order     
        sort_image_sim = np.sort(similarities[cap])[::-1]
        # Use highest position of actual pair's similarity as its rank
        rank = (np.where(sort_image_sim == correct_sim)[0][0] + 1)
        mrr += 1/(np.where(sort_image_sim == correct_sim)[0][0] + 1)

        if rank == 1:
            tp += 1
        else:
            fn += 1

    return mrr/len(similarities), tp, fn


def main(args, image_features, text_features):
    mrrs = []
    tp = 0  # true positives
    fn = 0  # false negatives

    for i in tqdm(range(0, len(image_features), args.eval_batch_size)):
        # Ignore batches that are not full
        if len(image_features) - i < args.eval_batch_size:
            continue

        sims = text_features @ image_features.T
        sims = sims.cpu().numpy()

        batch_mrr, batch_tp, batch_fn = evaluate(sims)
        mrrs.append(batch_mrr)
        tp += batch_tp
        fn += batch_fn
    
    recall = tp / (tp + fn + 1e-12)

    return recall, np.mean(mrrs), np.std(mrrs)


if  __name__ == "__main__":
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    print("Loading model ...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    image_features = []
    text_features = []
    if args.dataset_type == "coco":
        print("Fetching COCO data ...")
        coco = COCO(args.caption_file)
        coco_dicts = coco.anns

        images = []
        captions = []
        for n, sample in enumerate(tqdm(coco_dicts.values(), desc="creating encoded pairs")):
            filepath = glob(os.path.join(args.image_dir, str(sample['image_id']).zfill(12) + "*"))[0]
            images.append(preprocess(Image.open(filepath).convert("RGB")))
            captions.append(sample['caption'])

            if (n > 0 and n % args.batch_size == 0):
                batch_img_feats, batch_txt_feats = build_features(model, tokenizer, images, captions)
                image_features.append(batch_img_feats)
                text_features.append(batch_txt_feats)
                images, captions = [], []
        
        # last batch may not be processed
        if images:
            batch_img_feats, batch_txt_feats = build_features(model, tokenizer, images, captions)
            image_features.append(batch_img_feats)
            text_features.append(batch_txt_feats)
            del images, captions
    
    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    recalls_1 = []
    mean_mrrs = []
    std_mrrs = []
    for i in range(args.iterations):
        print(f"Evaluation iteration {i}")

        indices = torch.randperm(image_features.size(0), generator=torch.manual_seed(i))
        image_features, text_features = image_features[indices], text_features[indices]

        recall, mean_mrr, std_mrr = main(
            args=args,
            image_features=image_features,
            text_features=text_features
        )
        recalls_1.append(recall)
        mean_mrrs.append(mean_mrr)
        std_mrrs.append(std_mrr)

    print(f"Recall@1 / iter = {recalls_1}")
    print(f"Recall@1 Mean = {np.mean(recalls_1)}")
    print(f"Image MRRs (Batches of {args.eval_batch_size}) - Mean / iter: {mean_mrrs}, Std / iter: {std_mrrs}")
    print(f"Image MRRs (Batches of {args.eval_batch_size}) - Mean: {np.mean(mean_mrrs)}, Std: {np.mean(std_mrrs)}")
