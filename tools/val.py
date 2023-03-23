import torch
import argparse
import json
import yaml
import os
from tqdm import tqdm
from pathlib import Path
from tabulate import tabulate
from torch.utils.data import DataLoader
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

import sys

sys.path.insert(0, '.')
from imgcap.models import ClipCap
from imgcap.dataset import CaptionDataset
from imgcap.decode import generate_captions

from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap


def get_annotation_file(dataset: str):
    if dataset == 'COCO':
        return 'coco_caption/annotations/captions_val2014.json'
    elif dataset == 'Flickr8k':
        return ''
    elif dataset == 'Flickr30k':
        return ''
    elif dataset == 'ConceptualCaptions':
        return ''
    else:
        raise NotImplementedError


class Evaluator:
    def __init__(self, annot_path, results_path) -> None:
        coco = COCO(annot_path)
        cocoRes = coco.loadRes(results_path)
        self.cocoEval = COCOEvalCap(coco, cocoRes)

        # evaluate on a subset of images
        self.cocoEval.params['image_id'] = cocoRes.getImgIds()

    def evaluate(self):
        self.cocoEval.evaluate()

    def get_scores(self):
        return self.cocoEval.eval


def evaluate(model, dataloader, tokenizer, feat_len):
    results = []
    print("Starting Evaluation...")
    for _, _, img_feat, image_id, caption in tqdm(dataloader):
        with torch.inference_mode():
            # make data to gpu
            img_feat = img_feat.to('cuda:1', dtype=torch.float32)
            img_feat = img_feat.reshape(1, 512)  # kuhn edited
            img_embed = model.clip_project(img_feat).reshape(1, feat_len, -1)

        predicted_caption = generate_captions(model, tokenizer, img_embed)
        # print(f"Real Caption : {caption}\nPred Caption : {predicted_caption}\n")

        results.append({
            "image_id": image_id.item(),
            "caption": predicted_caption
        })
    return results


import os


def load_huggingface_model(model_class, model_name, local_file, **kwargs):
    if_loaded = True
    if os.path.exists(local_file):
        print(f"try Loading {model_class} from local file {local_file}")
        try:
            model = model_class.from_pretrained(local_file)
        except Exception as e:
            print(e)
            print("failed to load from local file, try to load from huggingface model")
            if_loaded = False
    else:
        if_loaded = False

    if not if_loaded:
        print(f"Loading {model_class} from huggingface model {model_name}")
        model = model_class.from_pretrained(model_name)
        model.save_pretrained(local_file)
        print(f"Saved {model_class} to local file {local_file}")
    if kwargs.get("return_tokenizer", None) == True:
        return model.tokenizer
    elif kwargs.get("return_feature_extractor", None) == True:
        return model.feature_extractor
    elif kwargs.get("return_vision_model", None) == True:
        return model.vision_model
    return model


def main(cfg, save_dir: Path):
    feat_len = cfg['SEQ_LENGTH']
    result_tmp_file = save_dir / 'results_test.json'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    LOCAL_TOKENIZER_PATH = '.cache/gpt2_tokenizer.pt'
    tokenizer = load_huggingface_model(GPT2Tokenizer, "gpt2", LOCAL_TOKENIZER_PATH)

    model = ClipCap(feat_len)
    model.load_state_dict(torch.load(cfg['WEIGHTS'], map_location=device))
    model = model.to(device)
    model.eval()

    dataset = CaptionDataset(cfg['VAL_DATA_PATH'], feat_len)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True)

    results = evaluate(model, dataloader, tokenizer, feat_len)

    print("Saving Results File...")
    with open(result_tmp_file, 'w') as f:
        json.dump(results, f)

    evaluator = Evaluator(get_annotation_file(cfg['DATASET']), result_tmp_file)
    evaluator.evaluate()
    scores = evaluator.get_scores()

    print(tabulate(list(scores.items()), numalign='left'))

    # os.remove(result_tmp_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)

    main(cfg, save_dir)
    #  python tools/val.py --cfg configs/defaults.yaml
