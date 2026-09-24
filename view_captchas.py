import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    RISKY_DATASET,
    GOLDEN_DATASET,
    MASTER_DATASET,
    TEACHER_MODEL_PATH,
    STUDENT_MODEL_PATH,
    FINAL_DEPLOY_MODEL,
    IMG_WIDTH,
    IMG_HEIGHT,
    NUM_CLASSES,
    CAPTCHA_LENGTH,
    IDX2CHAR,
    DEVICE,
)
from models import CaptchaAttentionCRNN, CaptchaStudentCNN

# Default Configuration
BATCH_SIZE = 4
RENDER_MODE = "half"    # "half" (compact 2 vertical px per cell) or "full" (full blocks ██)
USE_COLOR = True        # ANSI 24-bit TrueColor grayscale
INVERT_COLORS = True    # Invert light/dark polarity
CROP_PADDING = True     # Crop trailing zero-padded columns for display

# Target dataset selection
TARGET_FILE = RISKY_DATASET if os.path.exists(RISKY_DATASET) else (GOLDEN_DATASET if os.path.exists(GOLDEN_DATASET) else MASTER_DATASET)
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    TARGET_FILE = sys.argv[1]

# Grayscale density ramp for full-block rendering fallback
RAMP_FULL = ["  ", "··", "░░", "▒▒", "▓▓", "██"]


def load_inference_model():
    """
    Attempts to load a trained model (Teacher or Student) for live prediction and confidence scores.
    Returns: (model, model_name, device) or (None, None, None)
    """
    model = None
    model_name = None

    if os.path.exists(TEACHER_MODEL_PATH):
        try:
            m = CaptchaAttentionCRNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)
            m.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE))
            m.eval()
            return m, os.path.basename(TEACHER_MODEL_PATH), str(DEVICE)
        except Exception as e:
            print(f"[!] Warning: Failed to load teacher model: {e}")

    if os.path.exists(STUDENT_MODEL_PATH):
        try:
            m = CaptchaStudentCNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)
            m.load_state_dict(torch.load(STUDENT_MODEL_PATH, map_location=DEVICE))
            m.eval()
            return m, os.path.basename(STUDENT_MODEL_PATH), str(DEVICE)
        except Exception as e:
            print(f"[!] Warning: Failed to load student model: {e}")

    return None, "No Model Loaded (Using Dataset Labels)", "N/A"


def predict_batch(model, raw_images_batch):
    """
    Runs model inference on raw (17, 60) images.
    Returns: list of dicts with 'pred_text', 'avg_conf', 'char_confs'
    """
    if model is None:
        return None

    # Prepare tensor (Batch, 1, H, W)
    tensors = torch.FloatTensor(np.array(raw_images_batch))
    if tensors.ndim == 3:
        tensors = tensors.unsqueeze(1)
    tensors = tensors.to(DEVICE)

    with torch.no_grad():
        out = model(tensors)  # Supports both Teacher (out, attn) and Student (out)
        if isinstance(out, tuple):
            out = out[0]
        probs = F.softmax(out, dim=-1)
        max_probs, preds = probs.max(dim=-1)

    results = []
    for b in range(len(raw_images_batch)):
        pred_chars = [IDX2CHAR.get(idx.item(), '?') for idx in preds[b]]
        pred_text = "".join(pred_chars)
        char_confs = [float(p.item() * 100.0) for p in max_probs[b]]
        avg_conf = float(np.mean(char_confs))

        results.append({
            'pred_text': pred_text,
            'avg_conf': avg_conf,
            'char_confs': char_confs
        })

    return results


def process_image_to_matrix(img, invert=INVERT_COLORS, crop_padding=CROP_PADDING):
    """Normalize input tensor to 2D uint8 matrix, crop padding, and adjust contrast for display."""
    if len(img.shape) == 3:
        img = img.squeeze(0)

    # Crop trailing zero-padded columns before color inversion
    if crop_padding:
        non_zero_cols = np.where(img.any(axis=0))[0]
        if len(non_zero_cols) > 0:
            valid_width = non_zero_cols[-1] + 1
            img = img[:, :valid_width]

    if img.max() <= 1.0:
        matrix = (img * 255.0).astype(np.uint8)
    else:
        matrix = img.astype(np.uint8)

    # Ensure dark background by default
    if np.median(matrix) > 127:
        matrix = 255 - matrix

    if invert:
        matrix = 255 - matrix

    return matrix


def render_half_block(matrix, use_color=USE_COLOR):
    """
    Renders 2 vertical pixels per terminal cell using unicode upper half block (▀).
    Preserves 1:1 pixel aspect ratio on 2:1 character cells without downsampling.
    """
    h, w = matrix.shape
    lines = []

    for y in range(0, h, 2):
        row_top = matrix[y]
        row_bot = matrix[y + 1] if (y + 1 < h) else np.zeros(w, dtype=np.uint8)

        line_chars = []
        for x in range(w):
            t = int(row_top[x])
            b = int(row_bot[x])

            if use_color:
                line_chars.append(f"\033[38;2;{t};{t};{t}m\033[48;2;{b};{b};{b}m▀\033[0m")
            else:
                if t > 128 and b > 128:
                    line_chars.append("█")
                elif t > 128:
                    line_chars.append("▀")
                elif b > 128:
                    line_chars.append("▄")
                else:
                    line_chars.append(" ")

        lines.append("".join(line_chars))

    return lines


def render_full_block(matrix, use_color=USE_COLOR):
    """Renders 1 pixel per 2-character block (██)."""
    lines = []
    for row in matrix:
        chars = []
        for p in row:
            val = max(0, min(255, int(p)))
            idx = int((val / 256.0) * len(RAMP_FULL))
            char = RAMP_FULL[min(idx, len(RAMP_FULL) - 1)]
            if use_color:
                chars.append(f"\033[38;2;{val};{val};{val}m{char}\033[0m")
            else:
                chars.append(char)
        lines.append("".join(chars))
    return lines


def print_captcha_cli(matrix, dataset_label, pred_info, idx, total, mode=RENDER_MODE, use_color=USE_COLOR):
    """Prints a single captcha, its dataset ground-truth, and model prediction with confidence."""
    if mode == "half":
        lines = render_half_block(matrix, use_color=use_color)
        w_chars = matrix.shape[1]
    else:
        lines = render_full_block(matrix, use_color=use_color)
        w_chars = matrix.shape[1] * 2

    bar = "=" * max(w_chars, 50)
    sub_bar = "-" * max(w_chars, 50)

    print("\n" + bar)
    mode_label = f"half-block ({matrix.shape[1]}x{matrix.shape[0]})" if mode == "half" else f"full-block ({matrix.shape[1]}x{matrix.shape[0]})"
    print(f"Sample [{idx + 1}/{total}] | Size: {matrix.shape[1]}x{matrix.shape[0]} | Mode: {mode_label}")
    print(bar)

    for line in lines:
        print(line)

    print(sub_bar)

    # Format Prediction & Confidence
    if pred_info is not None:
        p_text = pred_info['pred_text']
        conf = pred_info['avg_conf']
        char_confs_str = " ".join([f"{c:.0f}%" for c in pred_info['char_confs']])

        # Check match against dataset label
        clean_target = dataset_label.split('_')[1] if dataset_label.startswith('MC_') else dataset_label
        match_str = "[MATCH]" if p_text == clean_target else "[MISMATCH]"

        print(f"Target: {dataset_label}  |  Pred: {p_text}  ({conf:.1f}% conf)  {match_str}")
        print(f"Char Confs: [{char_confs_str}]")
    else:
        print(f"Dataset Label: {dataset_label}")

    print(bar)


def main():
    global RENDER_MODE, INVERT_COLORS, USE_COLOR, CROP_PADDING

    if not os.path.exists(TARGET_FILE):
        print(f"Error: Dataset not found: {TARGET_FILE}")
        return

    # Load Model
    model, model_name, device_str = load_inference_model()

    print(f"Loading dataset : {TARGET_FILE}")
    print(f"Active Model    : {model_name} (Device: {device_str})")

    data = np.load(TARGET_FILE)
    images = data['images']
    labels = data['labels']
    total = len(images)

    print(f"Total samples   : {total} | Batch size: {BATCH_SIZE} | Render mode: {RENDER_MODE} | Auto-crop: {CROP_PADDING}\n")

    current_idx = 0

    while current_idx < total:
        end_idx = min(current_idx + BATCH_SIZE, total)

        print(f"\n--- Batch: [{current_idx + 1} - {end_idx} / {total}] ---")

        batch_imgs = images[current_idx:end_idx]
        batch_lbls = labels[current_idx:end_idx]

        # Run live model inference if model is loaded
        preds = predict_batch(model, batch_imgs)

        for i in range(len(batch_imgs)):
            global_idx = current_idx + i
            matrix = process_image_to_matrix(batch_imgs[i], invert=INVERT_COLORS, crop_padding=CROP_PADDING)
            label = str(batch_lbls[i]).strip()
            pred_info = preds[i] if preds is not None else None

            print_captcha_cli(matrix, label, pred_info, global_idx, total, mode=RENDER_MODE, use_color=USE_COLOR)

        prompt_info = (
            f"\n[Enter] Next | "
            f"[c] Crop ({CROP_PADDING}) | "
            f"[m] Mode ({RENDER_MODE}) | "
            f"[i] Invert | "
            f"[g <idx>] Jump | "
            f"[q] Quit > "
        )
        cmd = input(prompt_info).strip()

        if cmd.lower() == 'q':
            break
        elif cmd.lower() == 'c':
            CROP_PADDING = not CROP_PADDING
            print(f"Crop padding set to: {CROP_PADDING}")
        elif cmd.lower() == 'm':
            RENDER_MODE = "full" if RENDER_MODE == "half" else "half"
            print(f"Render mode set to: {RENDER_MODE}")
        elif cmd.lower() == 'i':
            INVERT_COLORS = not INVERT_COLORS
            print(f"Invert colors: {INVERT_COLORS}")
        elif cmd.lower().startswith('g '):
            try:
                target_page = int(cmd.split()[1]) - 1
                if 0 <= target_page < total:
                    current_idx = target_page
                else:
                    print(f"Invalid index: Must be between 1 and {total}.")
            except ValueError:
                print("Usage: g <index> (e.g. g 50)")
        else:
            current_idx = end_idx


if __name__ == "__main__":
    main()
