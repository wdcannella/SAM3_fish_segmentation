# SAM3 Fish Segmentation

Uses [SAM3](https://huggingface.co/facebook/sam3) to segment a fish from an image, identify fin landmarks, and draw filleting guidelines.

## What it does

- Segments the fish using a text prompt (e.g. `"catfish"`)
- Rotates the image so the fish's long axis is horizontal
- Estimates dorsal, pelvic, and caudal fin locations
- Saves an annotated output image with overlays and fin lines

## Setup

Accept the model terms at https://huggingface.co/facebook/sam3, then:

```bash
pip install -r requirements.txt
huggingface-cli login
```

### GPU (recommended)

Requires an NVIDIA GPU with CUDA. Install the CUDA-enabled version of PyTorch instead of the default:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with your CUDA version (check with `nvidia-smi`). Common values: `cu118`, `cu121`, `cu124`.

### CPU

No extra steps — the default `pip install -r requirements.txt` is sufficient. Expect long runtimes (several minutes per image).

## Usage

Edit the paths and prompt at the bottom of `sam3_fish_segmentation.py`:

```python
IMAGE_PATH  = "../fish_pics/your_image.jpg"
OUTPUT_PATH = "../fish_pics/your_image_segmented.jpg"
PROMPT      = "catfish"
```

Then run:

```bash
python sam3_fish_segmentation.py
```

The script automatically uses CUDA if available, otherwise falls back to CPU.
