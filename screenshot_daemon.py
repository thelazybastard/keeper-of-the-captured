# TODO: image daemon classifying screenshots based on data inside.
#  - use placeholder demo folder in pictures - done
#  - extract data and convert to readable format - done
#  - create list of description keywords found in screenshots

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

def load_image_contents():
    images = scan_main_directory()
    model, processor = load_models()

    for image in images:
        each_image = Image.open(image)

        inputs = processor(images=each_image, return_tensors="pt", padding=True)

        # generate image description
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=40) # type: ignore

        # convert to readable format
        image_description = processor.decode(generated_ids[0], skip_special_tokens=True)
        print("Description:", image_description)

load_image_contents()


