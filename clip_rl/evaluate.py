from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
import os
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch

from datasets import load_dataset

import open_clip

from pycocotools.coco import COCO

from data.data_utils import NEGATE


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
        choices=["coco", "vsr-random", "vsr-zeroshot"],
        default="coco",
        help="type of dataset to inform image and caption loading"
    )

    parser.add_argument(
        '--model_type',
        choices=["open_clip"],
        default="open_clip",
        help="which type of model to evaluate from: [open_clip]"
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
        '--checkpoint',
        type=str,
        help="model checkpoint path"
    )
    parser.add_argument(
        '--bert_model',
        type=str,
        default="bert-base-uncased",
        help="bert model that bert4clip uses"
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


def evaluate_mrr(similarities):
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


def evaluate_multi_cls(class_tp, class_fp, class_fn):
    recall, precision, f1 = {}, {}, {}

    # compute macros
    macro_recall, macro_precision = [], []
    for tp, fp, fn in zip(class_tp, class_fp, class_fn):
        macro_recall.append(tp / (tp + fn + 1e-12))
        macro_precision.append(tp / (tp + fp + 1e-12))
    
    recall['macro'] = np.mean(macro_recall)
    precision['macro'] = np.mean(macro_precision)
    f1['macro'] = 2 * ((recall['macro'] * precision['macro']) / (recall['macro'] + precision['macro']))

    # compute micros
    total_tp = np.sum(class_tp)
    total_fp = np.sum(class_fp)
    total_fn = np.sum(class_fn)
    recall['micro'] = total_tp / (total_tp + total_fn + 1e-12)
    precision['micro'] = total_tp / (total_tp + total_fp + 1e-12)
    f1['micro'] = 2 * ((recall['micro'] * precision['micro']) / (recall['micro'] + precision['micro']))

    return recall, precision, f1


def main_mrr(args, image_features, text_features):
    mrrs = []
    tp = 0  # true positives
    fn = 0  # false negatives

    for i in tqdm(range(0, len(image_features), args.eval_batch_size)):
        # Ignore batches that are not full
        if len(image_features) - i < args.eval_batch_size:
            continue

        sims = text_features @ image_features.T
        sims = sims.cpu().numpy()

        batch_mrr, batch_tp, batch_fn = evaluate_mrr(sims)
        mrrs.append(batch_mrr)
        tp += batch_tp
        fn += batch_fn
    
    recall = tp / (tp + fn + 1e-12)

    return recall, np.mean(mrrs), np.std(mrrs)


def main_cls(image_features, text_features, targets):
    """
    image_features: (N, D)
    text_features: (N, C, D)
    targets: (N)
    """
    n_classes = text_features.size(1)
    class_tp = [0] * n_classes  # true positives per class
    class_fp = [0] * n_classes  # false positives per class
    class_fn = [0] * n_classes  # false negatives per class

    preds = []
    for i in tqdm(range(len(image_features))):
        sims = image_features[i].unsqueeze(0) @ text_features[i].T
        preds.append(sims.max(dim=1).indices.item())

    preds = np.array(preds)
    targets = np.array(targets)

    accuracy = np.sum(preds == targets) / len(targets)

    for c in range(n_classes):
        true_indices = np.argwhere(targets == c)
        class_tp[c] = np.sum(preds[true_indices] == c)
        class_fp[c] = np.sum(preds == c) - class_tp[c]
        class_fn[c] = np.sum(targets == c) - class_tp[c]

    recall, precision, f1 = evaluate_multi_cls(class_tp, class_fp, class_fn)

    return accuracy, recall, precision, f1


def main(args):
    print("Loading model ...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    task = None

    image_features = []
    text_features = []
    targets = []
    if args.dataset_type == "coco":
        print("Fetching COCO data ...")
        task = "retrieval"

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

    elif args.dataset_type.startswith("vsr"):
        task = "classification"
        args.iterations = 1

        data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
        if args.dataset_type.endswith("random"):
            print("Fetching VSR's random test set ...")
            vsr_dataset = load_dataset("cambridgeltl/vsr_random", data_files=data_files)
        else:
            print("Fetching VSR's zeroshot test set ...")
            vsr_dataset = load_dataset("cambridgeltl/vsr_zeroshot", data_files=data_files)
        vsr_dataset = vsr_dataset['test']
        
        images = []
        false_captions = []
        true_captions = []
        for n, sample in enumerate(tqdm(vsr_dataset, total=len(vsr_dataset), desc="creating encoded pairs")):
            if "train" in sample['image_link']:
                data_dir = "coco_2017/coco_train2017"
            else:
                data_dir = "coco_2017/coco_val2017"
            filepath = os.path.join(args.image_dir, data_dir, sample['image'])
            images.append(preprocess(Image.open(filepath).convert("RGB")))

            relation_idx = sample['caption'].find(sample['relation'])
            if sample['label']:
                true_captions.append(sample['caption'])
                relation_len = len(sample['relation'])
                false_captions.append(sample['caption'][:relation_idx] + NEGATE[sample['relation']] + sample['caption'][relation_idx+relation_len:])
            else:
                # if target is false, using negation does not guarantee truth
                true_captions.append(sample['caption'][:relation_idx] + "not " + sample['caption'][relation_idx:])
                false_captions.append(sample['caption'])

            targets.append(sample['label'])  # label = 0 = False, label = 1 = True

            if (n > 0 and n % args.batch_size == 0):
                batch_img_feats, batch_txt_feats = build_features(model, tokenizer, images, false_captions + true_captions)
                image_features.append(batch_img_feats)
                batch_txt_feats = torch.cat(batch_txt_feats.unsqueeze(1).split(batch_txt_feats.size(0) // 2, dim=0), dim=1)
                text_features.append(batch_txt_feats)
                images, true_captions, false_captions = [], [], []
            
        # last batch may not be processed
        if images:
            batch_img_feats, batch_txt_feats = build_features(model, tokenizer, images, false_captions + true_captions)
            image_features.append(batch_img_feats)
            batch_txt_feats = torch.cat(batch_txt_feats.unsqueeze(1).split(batch_txt_feats.size(0) // 2, dim=0), dim=1)
            text_features.append(batch_txt_feats)
            del images, true_captions, false_captions

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)

    metrics = defaultdict(list)
    for i in range(args.iterations):
        print(f"Evaluation iteration {i}")
        print(f"Running evaluation for {task}")

        if task in ["retrieval"]:
            indices = torch.randperm(image_features.size(0), generator=torch.manual_seed(i))
            image_features, text_features = image_features[indices], text_features[indices]
        
        if task == "retrieval":
            recall, mean_mrr, std_mrr = main_mrr(
                args=args,
                image_features=image_features,
                text_features=text_features
            )
            metrics['recalls_1'].append(recall)
            metrics['mean_mrrs'].append(mean_mrr)
            metrics['std_mrrs'].append(std_mrr)
        elif task == "classification":
            accuracy, recall, precision, f1 = main_cls(
                image_features=image_features,
                text_features=text_features,
                targets=targets,
            )
            metrics['accuracy'].append(accuracy)
            metrics['recall'].append(recall)
            metrics['precision'].append(precision)
            metrics['f1'].append(f1)
        else:
            raise NotImplementedError("Evaluation task missing or not implemented.")

    if task == "retrieval":
        print(f"Recall@1 / iter = {metrics['recalls_1']}")
        print(f"Recall@1 Mean = {np.mean(metrics['recalls_1'])}")
        print(f"Image MRRs (Batches of {args.eval_batch_size}) - Mean / iter: {metrics['mean_mrrs']}, Std / iter: {metrics['std_mrrs']}")
        print(f"Image MRRs (Batches of {args.eval_batch_size}) - Mean: {np.mean(metrics['mean_mrrs'])}, Std: {np.mean(metrics['std_mrrs'])}")
    elif task == "classification":
        print(f"Accuracy = {metrics['accuracy'][0]}", sep="\n\n")
        print(f"Macro Average Recall = {metrics['recall'][0]['macro']}")
        print(f"Macro Average Precision = {metrics['precision'][0]['macro']}")
        print(f"Macro Average F1 = {metrics['f1'][0]['macro']}", sep="\n\n")
        print(f"Micro Average Recall = {metrics['recall'][0]['micro']}")
        print(f"Micro Average Precision = {metrics['precision'][0]['micro']}")
        print(f"Micro Average F1 = {metrics['f1'][0]['micro']}")


if  __name__ == "__main__":
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parse_args()

    main(args)

