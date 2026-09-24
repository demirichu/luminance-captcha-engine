import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from collections import Counter

from config import *
from models import CaptchaAttentionCRNN

# --- SETTINGS ---
MCT_BATCH_SIZE = 128
MC_PASSES = 8          # We will test the same image 8 times with different perturbations
ENTROPY_THRESHOLD = 0.25 # Uncertainty threshold (Low = Very confident)


def enable_dropout(model):
    """
    Leaves only Dropout layers active. 
    BatchNorms remain frozen, but the network reveals its "uncertainty" by dropping different neurons on each pass.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def run_mc_triage():
    print(f"Loading dataset: {MASTER_DATASET}")
    data = np.load(MASTER_DATASET)
    images = data['images']
    old_labels = data['labels']
    total_images = len(images)
    print(f"Found {total_images} images. Triage Starting...\n")

    # Load Model
    model = CaptchaAttentionCRNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)
    model.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE))
    model.eval()
    enable_dropout(model) # Enable Dropout!

    golden_imgs, golden_labels = [], []
    risky_imgs, risky_labels = [], []
    quarantine_imgs, quarantine_labels = [], []

    start_time = time.time()

    # Test-Time Augmentation (TTA) - Slight perturbation and noise
    tta_augmenter = T.Compose([
        T.RandomAffine(degrees=2, translate=(0.02, 0.02)), 
        T.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.015, 0.0, 1.0)) 
    ])
    
    label_dtype = f'U{CAPTCHA_LENGTH}'

    with torch.no_grad():
        for i in range(0, total_images, MCT_BATCH_SIZE):
            batch_imgs = images[i : i + MCT_BATCH_SIZE]
            batch_old_labels = old_labels[i : i + MCT_BATCH_SIZE]
            
            tensor_imgs = torch.FloatTensor(batch_imgs)
            if tensor_imgs.ndim == 3:
                tensor_imgs = tensor_imgs.unsqueeze(1)
            tensor_imgs = tensor_imgs.to(DEVICE)
            
            batch_preds_strings = [] 
            batch_entropies = []

            # 8 Different Passes (Monte Carlo)
            for _ in range(MC_PASSES):
                shaken_imgs = tta_augmenter(tensor_imgs)
                out, _ = model(shaken_imgs, temperature=0.8, return_attention=True) 
                
                probs = torch.softmax(out, dim=-1)
                preds = out.argmax(dim=-1) # (Batch, CAPTCHA_LENGTH)
                
                # Entropy Calculation
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1) # (Batch, CAPTCHA_LENGTH)
                mean_entropy = entropy.mean(dim=1).cpu().numpy() # Average entropy for each word
                
                string_preds = ["".join([IDX2CHAR.get(c.item(), '?') for c in seq]) for seq in preds]
                
                batch_preds_strings.append(string_preds)
                batch_entropies.append(mean_entropy)

            batch_entropies = np.array(batch_entropies)

            # Voting and Distribution Analysis
            for b_idx in range(len(batch_imgs)):
                img = batch_imgs[b_idx]
                old_label = str(batch_old_labels[b_idx]).strip()
                
                # 8 predictions and 8 entropy scores for this image
                predicted_strings = [batch_preds_strings[p_idx][b_idx] for p_idx in range(MC_PASSES)]
                item_entropies = batch_entropies[:, b_idx] 

                vote_counts = Counter(predicted_strings)
                most_common_pred, vote_freq = vote_counts.most_common(1)[0]
                
                # Average entropy of 8 attempts
                avg_entropy = np.mean(item_entropies)

                # --- 1. PURE (GOLDEN) CRITERIA ---
                # Found the same result in all 8 attempts AND is very confident (low entropy) AND matches old label (optional logic here ignores old_label)
                if vote_freq >= 7 and avg_entropy <= 0.45:
                    golden_imgs.append(img)
                    golden_labels.append(most_common_pred)
                
                # --- 2. RISKY / SCHIZOPHRENIC CRITERIA ---
                # Voting got stuck at 5 or 6, or entropy is high (Model is slightly confused)
                elif vote_freq >= 4:
                    new_label = f"MC_{most_common_pred}_ENT_{avg_entropy:.2f}"
                    risky_imgs.append(img)
                    risky_labels.append(new_label)
                
                # --- 3. TRASH (QUARANTINE) ---
                # Model agreed on less than 3 attempts in 8 trials (Completely unreadable)
                else:
                    new_label = f"TRASH_ENT_{avg_entropy:.2f}"
                    quarantine_imgs.append(img)
                    quarantine_labels.append(new_label)

            if i > 0 and i % (MCT_BATCH_SIZE * 20) == 0:
                print(f"Scanned: {i} / {total_images}...")

    # Save Results
    print(f"\nOperation completed in {(time.time() - start_time):.2f} seconds!")
    print("\n=== TRIAGE REPORT ===")
    print(f"True Golden (Pure) : {len(golden_imgs)}")
    print(f"Risky (Manual Check): {len(risky_imgs)}")
    print(f"Quarantine (Trash)  : {len(quarantine_imgs)}")

    print("\nPackaging files...")
    if golden_imgs:
        np.savez_compressed(GOLDEN_DATASET, images=np.array(golden_imgs, dtype=np.float32), labels=np.array(golden_labels, dtype=label_dtype))
    if risky_imgs:
        np.savez_compressed(RISKY_DATASET, images=np.array(risky_imgs, dtype=np.float32), labels=np.array(risky_labels, dtype='U64'))
    if quarantine_imgs:
        np.savez_compressed(QUARANTINE_DATASET, images=np.array(quarantine_imgs, dtype=np.float32), labels=np.array(quarantine_labels, dtype='U64'))
    print("Process finished! Time to review the Risky file with manual_label.py.")

if __name__ == "__main__":
    run_mc_triage()