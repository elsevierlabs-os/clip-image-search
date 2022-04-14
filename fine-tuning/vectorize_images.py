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
parser.add_argument("output_path", help="path to TSV file")
args = parser.parse_args()

if args.model_path == "baseline":
    model_path = OPENAI_CLIP_HF_HUBID
else:
    model_path = args.model_path

model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(OPENAI_CLIP_HF_HUBID)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

collator = ImageCaptionCollator(processor)

# :HACK: reuse the same code to generate vectors for other splits
#        unseen split has no captions, so we put in dummy captions so we can
#        reuse code here
# test_ds = ImageCaptionDataset(
#     os.path.join(IMAGE_DATA_DIR, "training", "CaptionPredictionTraining2017"),
#     os.path.join(IMAGE_DATA_DIR, "training", "CaptionPredictionTraining2017-Captions.csv"))
# test_ds = ImageCaptionDataset(
#     os.path.join(IMAGE_DATA_DIR, "validation", "ConceptDetectionValidation2017"),
#     os.path.join(IMAGE_DATA_DIR, "validation", "CaptionPredictionValidation2017-Captions.csv"))
# test_ds = ImageCaptionDataset(
#     os.path.join(IMAGE_DATA_DIR, "unseen", "CaptionPredictionTesting2017"),
#     os.path.join(IMAGE_DATA_DIR, "unseen", "CaptionPredictionTesting2017-Captions-dummy.txt"))
test_ds = ImageCaptionDataset(
    os.path.join(IMAGE_DATA_DIR, "test", "ConceptDetectionValidation2017"),
    os.path.join(IMAGE_DATA_DIR, "test", "CaptionPredictionValidation2017-Captions.csv"))
test_dl = DataLoader(test_ds, 
                     batch_size=BATCH_SIZE, 
                     num_workers=mp.cpu_count() - 1,
                     collate_fn=collator)

fvec = open(args.output_path, "w")
for bid, (batch, image_ids) in enumerate(test_dl):
    if bid % 100 == 0:
        print("... {:d} batches (of {:d}) vectors generated".format(bid, BATCH_SIZE))
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

print("... {:d} batches (of {:d}) vectors generated, COMPLETE".format(bid, BATCH_SIZE))
fvec.close()
