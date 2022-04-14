import argparse
import multiprocessing as mp
import numpy as np
import os
import torch
import time

from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader

from clip_dataset import ImageCaptionDataset, ImageCaptionCollator

DATA_DIR = "../data"
IMAGE_DATA_DIR = "../../ImageCLEF2017-CaptionPrediction"

OPENAI_CLIP_HF_HUBID = "openai/clip-vit-base-patch32"
K_VALUES = [1, 3, 5, 10, 20]
BATCH_SIZE = 64

EVAL_REPORT = os.path.join(DATA_DIR, "eval-report.tsv")

def compute_batch_mrrs(probs, k_values):
    ranks = np.argsort(-probs, axis=1)
    batch_mrrs = np.zeros((ranks.shape[0], len(k_values)))
    for i in range(ranks.shape[0]):
        mrr_at_k = []
        for j, k in enumerate(k_values):
            rank = np.where(ranks[i, 0:k] == i)[0]
            if rank.shape[0] == 0:
                # item not found in top k, don't add to MRR
                batch_mrrs[i, j] = 0
            else:
                # item found in top k, add dimension
                batch_mrrs[i, j] = 1.0 / (rank[0] + 1)

    return batch_mrrs


def write_report(fout, batch_mrr):
    for i in range(batch_mrr.shape[0]):
        k_scores = ["{:.5f}".format(x) for x in batch_mrr[i, :].tolist()]
        fout.write(",".join(k_scores) + "\n")


############################ main ############################

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="'baseline' for HF model or path to local model")
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
test_ds = ImageCaptionDataset(
    os.path.join(IMAGE_DATA_DIR, "test", "ConceptDetectionValidation2017"),
    os.path.join(IMAGE_DATA_DIR, "test", "CaptionPredictionValidation2017-Captions.csv"))
test_dl = DataLoader(test_ds, 
                     batch_size=BATCH_SIZE, 
                     num_workers=mp.cpu_count() - 1,
                     collate_fn=collator)
                    

model.eval()

start = time.time()
fout = open(EVAL_REPORT, "w")

for bid, (inputs, _) in enumerate(test_dl):
    if bid % 10 == 0:
        print("{:d} batches processed".format(bid))

    inputs.to(device)
    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    batch_mrrs = compute_batch_mrrs(probs.detach().cpu().numpy(), K_VALUES)
    write_report(fout, batch_mrrs)
    # break

elapsed = time.time() - start
print("{:d} batches processed, COMPLETE".format(bid))
print("elapsed time: {:.3f} s".format(elapsed))
fout.close()

mrr_scores = []
with open(EVAL_REPORT, "r") as frep:
    for line in frep:
        scores = np.array([float(x) for x in line.strip().split(',')])
        mrr_scores.append(scores)
mrr_scores = np.array(mrr_scores)
eval_scores = np.mean(mrr_scores, axis=0)
frep.close()

print(" | ".join(["{:.5f}".format(x) for x in eval_scores.tolist()]))

