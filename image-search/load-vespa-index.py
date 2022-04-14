import argparse
import os
import numpy as np
import requests

# DATA_DIR = "/home/ubuntu/CaptionPrediction"
DATA_SUBDIRS = ["training", "validation", "test"]

APP_NAME = "clip-demo"
SCHEMA_NAME = "image"
ENDPOINT = "http://localhost:8080/document/v1/{:s}/{:s}/docid/{:d}"


parser = argparse.ArgumentParser()
parser.add_argument("image_dir", help="path to folder containing training, validation and test images and captions")
parser.add_argument("vector_dir", help="path to folder containing vector TSV files")
args = parser.parse_args()


# scan directories and compose paths
image_paths = {}
for data_subdir in DATA_SUBDIRS:
    for image_folder_cand in os.listdir(os.path.join(args.image_dir, data_subdir)):
        # print(data_subdir, subdir_content)
        if os.path.isdir(os.path.join(args.image_dir, data_subdir, image_folder_cand)):
            for image_file in os.listdir(os.path.join(args.image_dir, data_subdir, image_folder_cand)):
                image_path = os.path.join(args.image_dir, data_subdir, image_folder_cand, image_file)
                image_id = image_file.replace(".jpg", "")
                # if image_id in image_paths:
                #     print("duplicate image:", image_file)
                image_paths[image_id] = image_path

print("# of image paths:", len(image_paths))

image_captions = {}
for data_subdir in DATA_SUBDIRS:
    for image_folder_cand in os.listdir(os.path.join(args.image_dir, data_subdir)):
        if image_folder_cand.find("-Captions") > -1:
            with open(os.path.join(args.image_dir, data_subdir, image_folder_cand), "r") as f:
                for line in f:
                    image_id, caption = line.strip().split('\t')
                    image_captions[image_id] = caption                

print("# of image captions:", len(image_captions))


doc_id = 1
failures, successes = 0, 0
headers = { "Content-Type": "application/json" }
for vec_file_cand in os.listdir(args.vector_dir):
    if vec_file_cand.startswith("vectors-"):
        with open(os.path.join(args.vector_dir, vec_file_cand), "r") as f:
            for line in f:
                image_id, vec_str = line.strip().split('\t')
                vec = np.array([float(x) for x in vec_str.split(',')])
                vec_norm = np.linalg.norm(vec, 2)
                vec /= vec_norm
                vec = vec.tolist()
                image_path = image_paths[image_id]
                try:
                    caption_text = image_captions[image_id]
                except KeyError:
                    caption_text = "(no caption provided)"
                input_rec = {
                    "fields": {
                        "image_id": image_id,
                        "image_path": image_path,
                        "caption_text": caption_text,
                        "clip_vector": {
                            "values": vec
                        }
                    }
                }
                url = ENDPOINT.format(APP_NAME, SCHEMA_NAME, doc_id)
                resp = requests.post(url, headers=headers, json=input_rec)
                if resp.status_code != 200:
                    print("ERROR loading [{:d}] {:s}: {:s}".format(
                        doc_id, image_id, resp.reason))
                    failures += 1
                else:
                    successes += 1
                print("Inserted document {:20s}, {:6d} ok, {:6d} failed, {:6d} total\r"
                    .format(image_id, successes, failures, doc_id), end="")
                doc_id += 1

print("\n{:d} documents read, {:d} succeeded, {:d} failed, COMPLETE"
    .format(doc_id + 1, successes, failures))
