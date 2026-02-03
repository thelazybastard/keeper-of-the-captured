from transformers import CLIPModel, CLIPProcessor, logging
import os
import warnings

os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()
logging.disable_progress_bar()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.save_pretrained("./models/clip-model")
processor.save_pretrained("./models/clip-processor")