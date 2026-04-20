# SAM3 Fish Segmentation

Uses [SAM3](https://huggingface.co/facebook/sam3) to segment a fish from an image, identify fin landmarks, and draw filleting guidelines.

## What it does

- Segments the fish using a text prompt (e.g. `"catfish"`)
- Rotates the image so the fish's long axis is horizontal
- Estimates dorsal, pelvic, and caudal fin locations
- Saves an annotated output image with overlays and fin lines

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login  # required to download SAM3 weights
```

## Usage

Edit the paths and prompt at the bottom of `sam3_fish_segmentation.py`, then run:

```bash
python sam3_fish_segmentation.py
```

## Notes

- Runs on CPU only (no CUDA required)
- SAM3 model weights are downloaded automatically on first run
- Accept the model terms at https://huggingface.co/facebook/sam3 before running
