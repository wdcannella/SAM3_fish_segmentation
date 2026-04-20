import sys
import types
import os

# Mock triton — not available on Windows
t = types.ModuleType('triton')
sys.modules['triton'] = t
tl = types.ModuleType('triton.language')
sys.modules['triton.language'] = tl
t.language = tl
t.jit = lambda x: x
tl.constexpr = int
tl.dtype = types.ModuleType('triton.language.dtype')
sys.modules['triton.language.dtype'] = tl.dtype

import time
import torch
import numpy as np
import cv2
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def patch_torch_for_cpu():
    """Redirect any CUDA tensor ops to CPU when CUDA is unavailable."""
    for name in ['zeros', 'ones', 'empty', 'arange', 'linspace', 'logspace', 'eye', 'full', 'tensor']:
        original = getattr(torch, name)
        def patched(*args, _orig=original, **kwargs):
            if 'device' in kwargs and kwargs['device'] == 'cuda' and not torch.cuda.is_available():
                kwargs['device'] = 'cpu'
            return _orig(*args, **kwargs)
        setattr(torch, name, patched)

    original_pin_memory = torch.Tensor.pin_memory
    def patched_pin_memory(self, *args, **kwargs):
        return self if self.device.type == 'cpu' else original_pin_memory(self, *args, **kwargs)
    torch.Tensor.pin_memory = patched_pin_memory


def cast_model_to_float32(model):
    """Cast all parameters/buffers to float32 and hook activations — required for CPU inference."""
    model.to(torch.float32)
    for module in model.modules():
        for name, buf in module.named_buffers(recurse=False):
            if buf.dtype in (torch.bfloat16, torch.float16):
                module.register_buffer(name, buf.to(torch.float32))
        for name, param in module.named_parameters(recurse=False):
            if param.dtype in (torch.bfloat16, torch.float16):
                param.data = param.data.to(torch.float32)

    def cast_hook(module, inputs):
        return tuple(
            x.to(torch.float32) if isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16 else x
            for x in inputs
        ) if isinstance(inputs, tuple) else inputs

    for module in model.modules():
        module.register_forward_pre_hook(cast_hook)

    return model


def estimate_fin_landmarks(contours):
    """
    Estimate the dorsal fin base and pelvic fin base from the fish mask contour.

    Strategy (works for any left/right-facing side-view fish):
      - Flatten all contour points into one array.
      - Compute the fish's horizontal (x) extent and centroid y.
      - Restrict to the middle 25–65 % of the fish's x range — this is where
        both fins appear in a typical side-profile catfish image.
      - Dorsal fin  = contour point with the smallest y in that x window
        (highest on the image = top of the back).
      - Pelvic fin  = contour point with the largest y in that x window
        (lowest on the image = underside near mid-body).

    Returns (dorsal_pt, pelvic_pt) as (x, y) integer tuples, or None if the
    contour is empty.
    """
    if not contours:
        return None, None

    pts = np.vstack([c.reshape(-1, 2) for c in contours])  # shape (N, 2)
    if len(pts) == 0:
        return None, None

    x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
    fish_width = x_max - x_min
    if fish_width == 0:
        return None, None

    # Middle band: 25 % to 65 % of the fish's body length
    band_lo = x_min + int(0.25 * fish_width)
    band_hi = x_min + int(0.65 * fish_width)

    in_band = pts[(pts[:, 0] >= band_lo) & (pts[:, 0] <= band_hi)]
    if len(in_band) == 0:
        return None, None

    dorsal_idx = np.argmin(in_band[:, 1])   # smallest y → highest on image
    pelvic_idx = np.argmax(in_band[:, 1])   # largest y  → lowest on image

    dorsal_pt = tuple(in_band[dorsal_idx].astype(int))
    pelvic_pt = tuple(in_band[pelvic_idx].astype(int))

    return dorsal_pt, pelvic_pt


def estimate_caudal_fin_start(mask):
    """
    Estimate where the caudal (tail) fin begins using the caudal peduncle —
    the narrow "neck" that connects the body to the tail.

    Strategy:
      - Scan every column in the rear 70–95 % of the fish's horizontal extent.
      - At each column, measure the vertical span of mask pixels (top_y to bottom_y).
      - The column with the minimum span is the caudal peduncle — the body
        pinches here before the tail fan opens outward again.
      - Return the top and bottom mask points at that column as the two
        endpoints of the caudal cut line.

    Args:
        mask: Boolean numpy array, shape (H, W).

    Returns:
        (top_pt, bot_pt) as (x, y) integer tuples, or (None, None).
    """
    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return None, None

    x_min, x_max = int(cols.min()), int(cols.max())
    fish_width = x_max - x_min
    if fish_width == 0:
        return None, None

    search_lo = x_min + int(0.70 * fish_width)
    search_hi = x_min + int(0.95 * fish_width)
    search_cols = [x for x in range(search_lo, search_hi + 1) if mask[:, x].any()]
    if not search_cols:
        return None, None

    best_x = min(search_cols, key=lambda x: mask[:, x].sum())

    col_ys = np.where(mask[:, best_x])[0]
    top_pt = (best_x, int(col_ys.min()))
    bot_pt = (best_x, int(col_ys.max()))

    return top_pt, bot_pt


def align_fish_horizontal(img_np, mask_uint8):
    """
    Rotate the image and mask so the fish's long axis lies along the x-axis.

    Uses second-order central moments (PCA of the mask) to find the principal
    axis angle, then rotates around the mask centroid.

    Returns (rotated_img, rotated_mask_uint8, angle_deg).
    """
    M = cv2.moments(mask_uint8)
    if M['m00'] == 0:
        return img_np, mask_uint8, 0.0

    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    mu20 = M['mu20'] / M['m00']
    mu02 = M['mu02'] / M['m00']
    mu11 = M['mu11'] / M['m00']
    angle_deg = np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02))

    h, w = img_np.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated_img = cv2.warpAffine(img_np, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    rotated_mask_raw = cv2.warpAffine(mask_uint8, rot_mat, (w, h), flags=cv2.INTER_NEAREST)
    rotated_mask_uint8 = (rotated_mask_raw > 127).astype(np.uint8) * 255

    return rotated_img, rotated_mask_uint8, angle_deg


def segment_fish(image_path, output_path, prompt="fish", confidence_threshold=0.2):
    """
    Run SAM3 text-prompted segmentation on a fish image.

    Args:
        image_path: Path to the input RGB image.
        output_path: Path to save the annotated output image.
        prompt: Text prompt describing the target object.
        confidence_threshold: Used only as a warning threshold for display purposes.
            All detections are kept so that degraded CPU scores don't suppress results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        patch_torch_for_cpu()

    t_total_start = time.perf_counter()

    print("Loading SAM3 model...")
    t_load_start = time.perf_counter()
    try:
        model = build_sam3_image_model(device=device)
        if device == "cpu":
            model = cast_model_to_float32(model)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Fix: accept terms at https://huggingface.co/facebook/sam3 then run: huggingface-cli login")
        return False
    t_load_end = time.perf_counter()
    print(f"Model load time: {t_load_end - t_load_start:.2f}s")

    if not os.path.exists(image_path):
        print(f"Error: image not found at {image_path}")
        return False

    # Use threshold=0.0 so the processor passes ALL detections through — CPU float32
    # inference degrades scores significantly and a hard threshold will drop everything.
    # We pick the best mask ourselves below.
    processor = Sam3Processor(model, device=device, confidence_threshold=0.0)

    print(f"Processing: {image_path}")
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)

    t_infer_start = time.perf_counter()
    state = {}
    with torch.inference_mode():
        processor.set_image(pil_img, state=state)
        processor.set_text_prompt(prompt, state=state)

    t_infer_end = time.perf_counter()
    print(f"Inference time:    {t_infer_end - t_infer_start:.2f}s")

    if "masks" not in state or state["masks"].numel() == 0:
        print(f"No detections returned for prompt '{prompt}'")
        return False

    scores = state["scores"]           # shape [N]
    masks = state["masks"]             # shape [N, 1, H, W], bool, thresholded at 0.5
    boxes = state["boxes"]             # shape [N, 4], pixel coords (x0, y0, x1, y1)

    print(f"Raw detection scores ({len(scores)} total): {[f'{s:.4f}' for s in scores.tolist()]}")

    # Pick the highest-confidence detection
    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()
    mask = masks[best_idx, 0].cpu().numpy()  # bool, H x W
    box = boxes[best_idx].cpu().numpy().astype(int)

    print(f"Best detection — score: {best_score:.4f}, mask coverage: {mask.sum()} px")
    if best_score < confidence_threshold:
        print(f"Warning: best score {best_score:.4f} is below threshold {confidence_threshold} (proceeding anyway)")

    # If the 0.5-threshold mask is empty (can happen with degraded CPU scores),
    # fall back to a lower threshold on the raw probability map
    if mask.sum() == 0:
        print("Mask empty at 0.5 threshold — falling back to 0.1 threshold on probability map")
        mask_probs = state["masks_logits"][best_idx, 0].cpu().numpy()  # already sigmoid'd by processor
        mask = mask_probs > 0.1
        print(f"Fallback mask coverage: {mask.sum()} px")

    # Largest connected component — removes stray mask pixels outside the fish
    mask_uint8 = (mask * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest)
        mask_uint8 = (mask * 255).astype(np.uint8)

    # Light morphological closing to smooth the mask boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask = mask_uint8 > 0

    # Rotate so the fish long axis lies along the x-axis before fin detection
    img_np, mask_uint8, rotation_angle = align_fish_horizontal(img_np, mask_uint8)
    mask = mask_uint8 > 0
    print(f"Aligned fish: rotated {rotation_angle:.1f}° so long axis is horizontal")

    # Visualization: semi-transparent red overlay + green contour + blue bounding box
    overlay = img_np.copy()
    overlay[mask] = (overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Recompute bounding box from the rotated contour
    if contours:
        rx, ry, rw, rh = cv2.boundingRect(np.vstack(contours))
        box = np.array([rx, ry, rx + rw, ry + rh])
    cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    # --- Filleting line: dorsal fin base → pelvic fin base ---
    dorsal_pt, pelvic_pt = estimate_fin_landmarks(contours)
    if dorsal_pt is not None and pelvic_pt is not None:
        print(f"Dorsal fin coordinates:  x={dorsal_pt[0]}, y={dorsal_pt[1]}")
        print(f"Pelvic fin coordinates:  x={pelvic_pt[0]}, y={pelvic_pt[1]}")

        # Yellow fillet line
        cv2.line(overlay, dorsal_pt, pelvic_pt, (255, 255, 0), 3, cv2.LINE_AA)

        # Magenta dots at each fin landmark
        cv2.circle(overlay, dorsal_pt, 6, (255, 0, 255), -1)
        cv2.circle(overlay, pelvic_pt, 6, (255, 0, 255), -1)

        # Labels
        cv2.putText(overlay, "dorsal", (dorsal_pt[0] + 8, dorsal_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, "pelvic", (pelvic_pt[0] + 8, pelvic_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
    else:
        print("Warning: could not estimate fin landmarks from contour")

    # --- Caudal fin start line (cyan) ---
    caudal_top, caudal_bot = estimate_caudal_fin_start(mask)
    if caudal_top is not None and caudal_bot is not None:
        print(f"Caudal fin start (top):  x={caudal_top[0]}, y={caudal_top[1]}")
        print(f"Caudal fin start (bot):  x={caudal_bot[0]}, y={caudal_bot[1]}")

        cv2.line(overlay, caudal_top, caudal_bot, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(overlay, caudal_top, 6, (0, 200, 200), -1)
        cv2.circle(overlay, caudal_bot, 6, (0, 200, 200), -1)
        cv2.putText(overlay, "caudal", (caudal_top[0] + 8, caudal_top[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        print("Warning: could not estimate caudal fin start from mask")

    label = f"{prompt} ({scores[best_idx]:.2f})"
    cv2.putText(overlay, label, (box[0], max(box[1] - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved to: {output_path}")

    t_total_end = time.perf_counter()
    print(f"\n--- Timing Summary ---")
    print(f"Model load:  {t_load_end - t_load_start:.2f}s")
    print(f"Inference:   {t_infer_end - t_infer_start:.2f}s")
    print(f"Total:       {t_total_end - t_total_start:.2f}s")
    return True


if __name__ == "__main__":
    IMAGE_PATH  = "../fish_pics/catfish_dataset/catfish_data_22/left_rgb.png"
    OUTPUT_PATH = "../fish_pics/Random/sam3_segmented.jpg"
    PROMPT      = "catfish"

    segment_fish(
        image_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
        prompt=PROMPT,
        confidence_threshold=0.2,
    )
