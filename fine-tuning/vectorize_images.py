import argparse
import multiprocessing as mp
import numpy as np
import os
import torch
import time

from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader

from clip_dataset import ImageCaptionDataset, ImageCaptionCollator

OPENAI_CLIP_HF_HUBID = "openai/clip-vit-base-patch32"
IMAGE_DATA_DIR = "../../ImageCLEF2017-CaptionPrediction"
BATCH_SIZE = 64

############################ main ############################

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="path to local model (or 'baseline' for OpenAI CLIP)")
parser.add_argument("output_dir", help="path to folder containing TSV files")
args = parser.parse_args()

if args.model_path == "baseline":
    model_path = OPENAI_CLIP_HF_HUBID
else:
    model_path = args.model_path

if os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(OPENAI_CLIP_HF_HUBID)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

collator = ImageCaptionCollator(processor)
datasets = {
    "training": ImageCaptionDataset(
        os.path.join(IMAGE_DATA_DIR, "training", "CaptionPredictionTraining2017"),
        os.path.join(IMAGE_DATA_DIR, "training", "CaptionPredictionTraining2017-Captions.csv")),
    "validation": ImageCaptionDataset(
        os.path.join(IMAGE_DATA_DIR, "validation", "ConceptDetectionValidation2017"),
        os.path.join(IMAGE_DATA_DIR, "validation", "CaptionPredictionValidation2017-Captions.csv")),
    "test": ImageCaptionDataset(
        os.path.join(IMAGE_DATA_DIR, "unseen", "CaptionPredictionTesting2017"),
        os.path.join(IMAGE_DATA_DIR, "unseen", "CaptionPredictionTesting2017-Captions-dummy.txt"))
}
for split, dataset in datasets.items():
    test_dl = DataLoader(dataset, 
                        batch_size=BATCH_SIZE, 
                        num_workers=mp.cpu_count() - 1,
                        collate_fn=collator)

    fvec = open(
        os.path.join(args.output_dir, "vectors-{:s}.tsv".format(split)),
        "w")
    for bid, (batch, image_ids) in enumerate(test_dl):
        if bid % 100 == 0:
            print("... {:d} batches (of {:d}) vectors generated for {:s}".format(
                bid, BATCH_SIZE, split))
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.get_image_features(pixel_values=batch["pixel_values"])
            outputs = outputs.cpu().numpy()
        for i in range(outputs.shape[0]):
            image_id = image_ids[i]
            vector = outputs[i].reshape(-1).tolist()
            fvec.write("{:s}\t{:s}\n".format(
                image_id,
                ",".join(["{:.5f}".format(v) for v in vector])))        
        # break

    print("... {:d} batches (of {:d}) vectors generated for {:s}, COMPLETE".format(
        bid, BATCH_SIZE, split))
    fvec.close()
