# The Keeper of the Captured

AI powered image sorter that automatically categorizes and sorts your photos into folders based on their content, not names.

## Purpose

Keeper of the Captured scans a directory of your choice, analyzes each image using OpenAI's CLIP model, and organizes them into categorized folders based on what's in the picture. Perfect for when you have hundreds or thousands of images but don't want to go on a manhunt for a single picture. 

## Features

- Uses CLIP to read image contents
- Downloads the AI model once, then runs offline
- Preview mode, to show you which image will go where before commitment
- Creates organized folders based on image content
- Sorts images via keywords, ranging from basic desktop screenshots to professional photography
- Currently only supports JPG, JPEG, PNG, GIF, BMP, and WebP 

## Installation and Usage (non developers)

Download the executable file - no Python installation or dependencies required!

1. Download `keeper-of-the-captured.7z` from releases
2. Extract the contents to your desired location
3. Double click on keeper.exe
4. **Note:** Currently it takes a while to startup, so don't panic if nothing shows up on screen. Give it a while to load.

## For Developers

If you want to run from source or modify the code:

**Requirements:**
- Python 3.13+ 
- Required packages:
  ```bash
  pip install pillow transformers torch
  ```

**Setup:**
1. Clone this repo
2. Install dependencies
3. Ensure `image_keywords.py` is in the same directory as `keeper_of_the_captured.py`
4. Run with `python keeper_of_the_captured.py`

### Prerequisites

- Windows 10 or 11 operating systems

## Usage (for developers)

1. Run the script:
   ```bash
   python keeper_of_the_captured.py
   ```

2. Enter directory name (Must be in Users home directory i.e Videos, Music, Downloads, etc):
   ```
   Pictures/Screenshots
   ```

3. On first run, you'll be asked if you have a HuggingFace token:
   - If you have one, paste it to avoid warnings and increase download speed
   - If not, press 'n' to continue (the script will still work)

4. The script will download the CLIP model. around 600MB of free space needed

5. Review the preview showing where each image will be moved

6. Confirm with 'y' to proceed or 'n' to cancel

## Example

```
Enter directory name (Must be in Users home directory i.e Videos, Music, Downloads, etc): Pictures
Scan complete!
beach_sunset.jpg will be moved to Nature
family_photo.jpg will be moved to People
car_show.jpg will be moved to Vehicles
Proceed with cleanup? (y/n): y
beach_sunset.jpg has been moved to Nature
family_photo.jpg has been moved to People
car_show.jpg has been movedd to Vehicles
Cleanup complete!
```

## Future Updates

- Better optimization
- Linux support
- MacOS support
- Daemonized (run in background)

## Contributing

Pull requests are welcome! Feel free to open an issue if you find bugs or have suggestions.

## Author

Monish Giani (thelazybastard)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Built with:
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Pillow (PIL)](https://python-pillow.org/)
