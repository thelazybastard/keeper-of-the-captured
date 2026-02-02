# TODO: image daemon classifying screenshots based on data inside.
#  - use placeholder demo folder in pictures - done
#  - extract data and convert to readable format - done
#  - create list of keywords found in screenshots description - done
#  - create folder names based on image description
#  - create folders based on image description

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import torch
from image_keywords import all_keywords, keyword_to_folder
import shutil

screenshots = Path.home() / "Pictures/demo"

def load_models():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def scan_main_directory():
    for image in screenshots.iterdir():
        yield image

def load_image_contents():
    images = scan_main_directory()
    model, processor = load_models()
    labels = all_keywords

    for image in images:
        each_image = Image.open(image)

        inputs = processor(text=labels, images=each_image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        keyword_probability = outputs.logits_per_image.softmax(dim=1)[0]
        predicted_keyword = labels[keyword_probability.argmax()]
        folder_name = keyword_to_folder[predicted_keyword]
        return folder_name


load_image_contents()


