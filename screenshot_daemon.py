# TODO: image daemon classifying screenshots based on data inside.
#  - use placeholder demo folder in pictures - done

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import torch

screenshots = Path.home() / "Pictures/demo"

def load_models():
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    return model, processor


def scan_main_directory():
    for image in screenshots.iterdir():
        yield image





