"""
Generic CAPTCHA Dataset Formatter

Converts a folder of labeled CAPTCHA images into the canonical .npz training format.
Images should be named with their label, e.g.: AB7K.png, QP9M.png

Usage:
    python dataset_formatter.py --input ./raw --output ./files/dataset.npz

The formatter reads images, validates dimensions and labels against config.py,
converts to grayscale/luminance, normalizes to [0, 1], and saves as .npz.
"""
import os
import argparse
import hashlib
import numpy as np
from PIL import Image

from config import IMG_WIDTH, IMG_HEIGHT, CAPTCHA_LENGTH, CHARS, CHAR2IDX


def format_dataset(input_dir, output_path, label_from_filename=True):
    supported_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    files = [f for f in os.listdir(input_dir) 
             if os.path.splitext(f)[1].lower() in supported_exts]
    
    if not files:
        print(f"[!] No supported image files found in: {input_dir}")
        return
    
    print(f"[*] Found {len(files)} image files in {input_dir}")
    print(f"[*] Expected: {IMG_WIDTH}x{IMG_HEIGHT} grayscale, {CAPTCHA_LENGTH} chars from \"{CHARS}\"")
    
    images = []
    labels = []
    seen_hashes = set()
    
    skipped_dim = 0
    skipped_label = 0
    skipped_dup = 0
    skipped_error = 0
    
    for filename in sorted(files):
        filepath = os.path.join(input_dir, filename)
        
        # Extract label from filename (stem without extension)
        name_stem = os.path.splitext(filename)[0]
        if label_from_filename:
            label = name_stem.split('_')[0]  # Support "ABCD_extra.png" format
        else:
            label = name_stem
        
        # Validate label
        if len(label) != CAPTCHA_LENGTH:
            skipped_label += 1
            continue
        if not all(c in CHAR2IDX for c in label):
            skipped_label += 1
            continue
        
        # Load and validate image
        try:
            img = Image.open(filepath).convert('L')  # Convert to grayscale
        except Exception:
            skipped_error += 1
            continue
        
        w, h = img.size
        if w != IMG_WIDTH or h != IMG_HEIGHT:
            skipped_dim += 1
            continue
        
        # Normalize to [0, 1] float32
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        
        # Check for finite values
        if not np.all(np.isfinite(arr)):
            skipped_error += 1
            continue
        
        # Duplicate detection via hash
        img_hash = hashlib.md5(arr.tobytes()).hexdigest()
        if img_hash in seen_hashes:
            skipped_dup += 1
            continue
        seen_hashes.add(img_hash)
        
        images.append(arr)
        labels.append(label)
    
    # Report
    print(f"\n--- FORMATTER REPORT ---")
    print(f"Valid samples  : {len(images)}")
    print(f"Skipped (dims) : {skipped_dim}")
    print(f"Skipped (label): {skipped_label}")
    print(f"Skipped (dupe) : {skipped_dup}")
    print(f"Skipped (error): {skipped_error}")
    
    if len(images) == 0:
        print("[!] No valid samples to save.")
        return
    
    label_dtype = f'U{CAPTCHA_LENGTH}'
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(
        output_path,
        images=np.array(images, dtype=np.float32),
        labels=np.array(labels, dtype=label_dtype)
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[+] Dataset saved: {output_path} ({file_size_mb:.2f} MB)")
    print(f"[+] Shape: ({len(images)}, {IMG_HEIGHT}, {IMG_WIDTH})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert labeled CAPTCHA images to .npz dataset")
    parser.add_argument("--input", required=True, help="Input directory with labeled images")
    parser.add_argument("--output", default="files/dataset.npz", help="Output .npz file path")
    args = parser.parse_args()
    
    format_dataset(args.input, args.output)
