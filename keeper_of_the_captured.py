# TODO: image daemon classifying images based on data inside.
#  - use placeholder demo folder in pictures - done
#  - extract data and convert to readable format - done
#  - create list of keywords found in screenshots description - done
#  - create folder names based on image description - done
#  - create folders based on image description - done
#  - add recursive scanning - done
#  - add error handling - done
#  - add dry run - done
#  - hide model process in terminal - done
#  - create a seperate thread for CLIP loading
#  - turn into windows taskbar

from PIL import Image
from transformers import CLIPProcessor, CLIPModel, logging
from pathlib import Path
import torch
from image_keywords import all_keywords, keyword_to_folder
import shutil
import os
import warnings

os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
logging.disable_progress_bar()

user_input = input("Enter directory name (Must be in Users home directory i.e Videos, Music, Downloads, etc): ")
pictures = Path.home() / user_input

def ask_for_token():
    response = input("Do you have a HuggingFace token? (y/n): ").strip().lower()
    if response.lower() == "y":
            token = input("Paste your token: ").strip()
            if token:
                os.environ["HF_TOKEN"] = token
            else:
                print("No token entered, continuing without authentication.")
    else:
        print("Continuing without token. You may see a warning (this is normal).")


def load_models():
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    except Exception as e:
        print(f"Failed to load models: {e}")
        return None, None

def scan_main_directory():
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    try:
        for image in pictures.rglob('*'):
            if image.is_file() and image.suffix in image_extensions:
                yield image
    except (OSError, PermissionError) as e:
        print(f"Error scanning directory {pictures}: {e}")

def load_image_contents_and_sort(preview_mode=True):
    images = scan_main_directory()
    model, processor = load_models()
    labels = all_keywords

    if model is None:
        print("Could not load models, exiting...")
        exit()

    for image in images:
        try:
            each_image = Image.open(image)

            inputs = processor(text=labels, images=each_image, return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = model(**inputs)
                keyword_probability = outputs.logits_per_image.softmax(dim=1)[0]
                predicted_keyword = labels[keyword_probability.argmax()]
                folder_name = keyword_to_folder[predicted_keyword]

                if preview_mode:
                    for img in images:
                        print(f"{img.name} will be moved to {folder_name}")
                elif not preview_mode:
                    folder_path = pictures / folder_name
                    folder_path.mkdir(parents=True, exist_ok=True)
                    shutil.move(image, folder_path)

        except (IOError, OSError) as e:
            print(f"Failed to process {image.name}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error with {image.name}: {e}")
            continue



def preview_then_execute():
    ask_for_token()
    load_image_contents_and_sort(preview_mode=True)
    confirmation = input("Proceed with cleanup? (y/n): ")

    if confirmation.lower() == "y":
        load_image_contents_and_sort(preview_mode=False)
    elif confirmation.lower() == "n":
        print("Come back soon! The Keeper is always at your service")


preview_then_execute()




