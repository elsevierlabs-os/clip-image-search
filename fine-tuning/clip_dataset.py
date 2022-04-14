import os
import torch

from PIL import Image
from torch.utils.data import Dataset


class ImageCaptionDataset(Dataset):
    def __init__(self, image_folder, caption_file):

        super().__init__()
        self.image_folder = image_folder
        self.caption_file = caption_file
    
        self.image_to_caption = {}
        self.images = []
        with open(self.caption_file, "r") as fcap:
            for line in fcap:
                image_id, caption = line.strip().split('\t')
                if os.path.exists(os.path.join(self.image_folder, image_id + ".jpg")):
                    self.image_to_caption[image_id] = caption
                    self.images.append(image_id)

    def __len__(self):
        return len(self.image_to_caption)

    def __getitem__(self, idx):
        image = self._get_image(idx)
        caption = self._get_caption(idx)
        return {
            "image_id": self.images[idx],
            "image": image,
            "caption": caption
        }

    def _get_image(self, idx):
        image_id = self.images[idx]
        image = Image.open(os.path.join(self.image_folder, image_id + ".jpg"))
        image = image.convert("RGB")
        return image

    def _get_caption(self, idx):
        image_id = self.images[idx]
        caption = self.image_to_caption[image_id]
        return caption


class ImageCaptionCollator(object):
    def __init__(self, processor, 
                 image_size=224,
                 max_caption_length=64):
        self.processor = processor
        self.image_size = image_size
        self.max_caption_length = max_caption_length

    def __call__(self, batch):
        image_ids = [row["image_id"] for row in batch]
        images = [row["image"] for row in batch]
        captions = [row["caption"] for row in batch]
        # image preprocessing: feature extractor defaults
        # caption preprocessing: pad/truncate + tensor
        inputs = self.processor(text=captions,
                                images=images,
                                return_tensors="pt",
                                padding="max_length",
                                max_length=64,
                                truncation=True)
        return inputs, image_ids



# from transformers import CLIPProcessor, CLIPModel
# from torch.utils.data import DataLoader

# DATA_DIR = "../ImageCLEF2017-CaptionPrediction"

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# collator = ImageCaptionCollator(processor)
# train_ds = ImageCaptionDataset(
#     os.path.join(DATA_DIR, "training", "CaptionPredictionTraining2017"),
#     os.path.join(DATA_DIR, "training", "CaptionPredictionTraining2017-Captions.csv"))
# train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0,
#                      collate_fn=collator)


# for bid, (inputs, _) in enumerate(train_dl):
#     inputs.to(device)
#     outputs = model(**inputs)

#     logits_per_image = outputs.logits_per_image
#     probs = logits_per_image.softmax(dim=1)
#     print(probs)
#     break
