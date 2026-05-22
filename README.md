# The Keeper of the Captured

An AI-powered image sorter that automatically categorizes and sorts your photos into folders based on their content, not names.

## Purpose

Keeper of the Captured scans a directory of your choice, analyzes each image using a local instance of the **Gemma 3 (4B)** model via **Ollama**, and organizes them into categorized folders based on what's in the picture. Perfect for when you have hundreds or thousands of images but don't want to go on a manhunt for a single picture.

## Features

- Uses a localized Vision Language Model (Gemma 3 4B) to read image contents directly on your machine.
- Automatically handles downloading and pulling the required AI model via Ollama.
- Runs entirely offline — no data is sent to external APIs.
- Preview mode to show you which image will go where before commitment.
- Creates organized folders in your system's `Pictures` directory based on image content.
- Supports JPG, JPEG, PNG, BMP, and WebP formats.
- Cross-platform support (Windows, macOS, Linux).

## Installation and Usage

To run this tool, you must have [Ollama](https://ollama.com/) installed on your machine.

### 1. Install Ollama

Download and install Ollama from [their official website](https://ollama.com/download). Ensure the `ollama` command is available in your terminal.

### 2. Download and Setup

Download the standalone executable for your operating system from the [Releases](https://github.com/thelazybastard/keeper-of-the-captured/releases/latest) page.

To make the tool easily accessible from anywhere in your terminal, it is recommended to rename the file and add it to your system `PATH`:

**For Linux / macOS:**

Open your terminal and run the following commands (replace `keeper-linux-x64` with the file you downloaded):

```bash
# Make the downloaded binary executable
chmod +x keeper-linux-x64

# Rename and move it to your system PATH
sudo mv keeper-linux-x64 /usr/local/bin/keeper
```

Now you can simply type `keeper` in any terminal to run the program!

**For Windows:**

1. Download `keeper-windows-x64.exe`.
2. Rename the file to `keeper.exe`.
3. Move `keeper.exe` to a permanent folder (e.g., `C:\Program Files\Keeper\`).
4. Add that folder to your System `PATH`:
   - Open Start and search for "Environment Variables".
   - Click "Edit the system environment variables".
   - Select "Environment Variables", find "Path", and click "Edit".
   - Click "New", add the path to the folder containing `keeper.exe`, and save.
5. You can now launch it by typing `keeper` in Command Prompt or PowerShell.

## Usage

1. Open your terminal or command prompt and run the tool:

   ```bash
   keeper
   ```

2. On the first run, the tool will automatically check if the `gemma3:4b` model is downloaded via Ollama. If it isn't, it will trigger the download (this requires a few GBs of free space and may take a while depending on your internet connection).

3. Enter the directory name you want to clean. **Note:** This directory path is relative to your Home folder (e.g., `Downloads/UnsortedImages` or `Desktop/MessyFolder`).

4. The script will analyze the images and print a preview of their new destinations.

5. Confirm with `y` to proceed or `n` to cancel. The images will be moved into categorized folders within your system's `Pictures` directory.

## Example

```text
Checking for required model (gemma3:4b)...
Model 'gemma3:4b' is available.
Enter Directory to clean: Downloads/Photos

Image: "/home/user/Downloads/Photos/beach_sunset.jpg"
Destination(in Pictures USER folder): Nature & Landscapes

Image: "/home/user/Downloads/Photos/desktop_ss.png"
Destination(in Pictures USER folder): Screenshots

Proceed with operation? (y/n): y
Moved "/home/user/Downloads/Photos/beach_sunset.jpg" -> "Nature & Landscapes"
Moved "/home/user/Downloads/Photos/desktop_ss.png" -> "Screenshots"
```

## For Developers

If you want to run from source or modify the code:

**Requirements:**
- [Rust / Cargo](https://rustup.rs/) toolchain installed.
- [Ollama](https://ollama.com/) installed.

**Setup:**

1. Clone this repo:

   ```bash
   git clone https://github.com/thelazybastard/keeper-of-the-captured.git
   cd keeper-of-the-captured
   ```

2. Build and run the project using Cargo:

   ```bash
   cargo run --release
   ```

## Future Updates

- Better optimization
- Daemonized (run in background / automated watch folders)
- Custom model selection

## Contributing

Pull requests are welcome! Feel free to open an issue if you find bugs or have suggestions.

## Author

Monish Giani (thelazybastard)

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Acknowledgements

Built with:
- [Ollama](https://ollama.com/)
- [Gemma Model by Google](https://deepmind.google/technologies/gemma/)
- [Rust](https://www.rust-lang.org/)
