import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import kornia.augmentation as K

from config import *
from models import CaptchaAttentionCRNN
import copy

def pgd_attack(model, images, labels, epsilon, alpha, iters, criterion, temp, n_restarts=1):
    """PGD with multi-restart and per-sample worst-case selection.
    Batch-average selection would miss worst-case for individual samples;
    per-sample selection keeps the highest-loss perturbation for each image."""
    was_training = model.training
    model.train()

    images = images.clone().detach().to(DEVICE)
    labels = labels.to(DEVICE)
    B = images.size(0)

    best_perturbed = None
    best_loss_ps = None  # per-sample loss tracker (B,)

    _ce_none = nn.CrossEntropyLoss(reduction='none')  # for per-sample selection

    for _ in range(n_restarts):
        perturbed = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        perturbed = torch.clamp(perturbed, 0, 1)

        for i in range(iters):
            perturbed = perturbed.detach()
            perturbed.requires_grad_(True)

            # Full FP32 — FP16 gradients are less precise, produce weaker attacks.
            out, _ = model(perturbed, temperature=temp)
            loss = criterion(out.view(-1, NUM_CLASSES), labels.view(-1))

            grad = torch.autograd.grad(loss, perturbed, only_inputs=True)[0]

            with torch.no_grad():
                perturbed = perturbed + alpha * grad.sign()
                eta = torch.clamp(perturbed - images, min=-epsilon, max=epsilon)
                perturbed = torch.clamp(images + eta, min=0, max=1)

        # Per-sample selection: keep whichever restart was worst for EACH sample.
        # Avoids batch-average masking individual worst-cases.
        with torch.no_grad():
            out_eval, _ = model(perturbed, temperature=temp)
            loss_ps = _ce_none(out_eval.view(-1, NUM_CLASSES),
                               labels.view(-1)).view(B, -1).mean(dim=1)  # (B,)

        if best_perturbed is None:
            best_perturbed = perturbed.detach()
            best_loss_ps = loss_ps
        else:
            # Where this restart found a worse perturbation, use it
            better = (loss_ps > best_loss_ps).view(B, 1, 1, 1)
            best_perturbed = torch.where(better, perturbed.detach(), best_perturbed)
            best_loss_ps = torch.maximum(loss_ps, best_loss_ps)

    if was_training:
        model.train()

    return best_perturbed

class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        self.ema_model.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema_model.state_dict()
            for key in esd:
                esd[key].copy_(esd[key] * self.decay + msd[key] * (1.0 - self.decay))

class CustomCaptchaDataset(Dataset):
    def __init__(self, images, labels, is_hard=None):
        self.images = images
        self.labels = labels
        self.is_hard = is_hard if is_hard is not None else np.zeros(len(images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label_str = str(self.labels[idx]).strip()
        if img.ndim == 2:
            tensor = torch.FloatTensor(img).unsqueeze(0)
        elif img.ndim == 3 and img.shape[0] in (1, 3):
            tensor = torch.FloatTensor(img)
        elif img.ndim == 3 and img.shape[2] in (1, 3):
            tensor = torch.FloatTensor(img).permute(2, 0, 1)
        else:
            tensor = torch.FloatTensor(img).unsqueeze(0)
        targets = [CHAR2IDX[c] for c in label_str]
        pseudo_filename = f"index_{idx}_{label_str}"
        return tensor, torch.LongTensor(targets), pseudo_filename, idx

def train():
    print("Loading Datasets...")
    dataset_path = GOLDEN_DATASET if os.path.exists(GOLDEN_DATASET) else MASTER_DATASET
    golden_data = np.load(dataset_path)
    print(f"Dataset Loaded: {dataset_path}")
    raw_images, raw_labels = golden_data['images'], golden_data['labels']

    # FIX: pre-filter bad labels ONCE at load time — avoids recursive __getitem__
    # fallback that can silently oversample neighboring samples in a bad-label cluster.
    valid_mask = np.array([
        len(str(l).strip()) == CAPTCHA_LENGTH and all(c in CHAR2IDX for c in str(l).strip())
        for l in raw_labels
    ])
    if valid_mask.sum() < len(raw_labels):
        print(f"[!] Filtered {(~valid_mask).sum()} invalid labels from dataset.")
    final_images = raw_images[valid_mask]
    final_labels = raw_labels[valid_mask]
    hard_flags = np.zeros(len(final_images))

    if FINE_TUNE_MODE and os.path.exists(HARD_DATASET):
        print("Ultra-Hard Dataset added (will be weighted via Sampler)...")
        hard_data = np.load(HARD_DATASET)
        h_imgs, h_labels = hard_data['images'], hard_data['labels']
        final_images = np.concatenate([final_images, h_imgs], axis=0)
        final_labels = np.concatenate([final_labels, h_labels], axis=0)
        hard_flags = np.concatenate([hard_flags, np.ones(len(h_imgs))], axis=0)

    full_dataset = CustomCaptchaDataset(final_images, final_labels, hard_flags)

    model = CaptchaAttentionCRNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)

    if FINE_TUNE_MODE and os.path.exists(TEACHER_MODEL_PATH):
        model.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE))
        print(f"\n[!!!] EXISTING MODEL LOADED: {TEACHER_MODEL_PATH}\n")

    train_size = int(0.90 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, test_ds = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    # --- FIXED WEIGHTED SAMPLER ---
    # Performance/bug issue resolved by iterating only over indices in train_ds
    sample_weights = [3.0 if full_dataset.is_hard[idx] == 1 else 1.0 for idx in train_ds.indices]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)

    # 50k+ dataset: num_workers=4 + prefetch_factor=2 keeps RX 6600 fully fed.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4 if USE_CUDA else 0, pin_memory=USE_CUDA,
                              persistent_workers=True if USE_CUDA else False,
                              prefetch_factor=2 if USE_CUDA else None)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4 if USE_CUDA else 0, pin_memory=USE_CUDA,
                              persistent_workers=True if USE_CUDA else False,
                              prefetch_factor=2 if USE_CUDA else None)
    
    # --- AUGMENTATION: Fully GPU-native Kornia pipeline (zero CPU loops) ---
    augmenter = nn.Sequential(
        K.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.92, 1.08),
                       shear=(-4, 4), p=0.9),
        K.RandomPerspective(distortion_scale=0.12, p=0.25),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5), p=0.25),
        # K.RandomErasing replaces the CPU _random_erase loop — fully GPU-native
        K.RandomErasing(scale=(0.01, 0.08), ratio=(0.5, 2.0), value=0.0, p=0.30),
    ).to(DEVICE)
    # Elastic is separate: its probability decays with epoch (hurts at convergence)
    _k_elastic = K.RandomElasticTransform(
        kernel_size=(15, 15), sigma=(3.0, 6.0), alpha=(0.5, 1.5)
    ).to(DEVICE)

    def augmenter_fn(imgs, epoch):
        imgs = augmenter(imgs)
        # Per-sample elastic: Kornia's p applies independently to each image.
        # Previous: `if random.random() < p` applied elastic to ALL or NONE of batch.
        _k_elastic.p = float(0.15 * np.exp(-epoch / 20))
        imgs = _k_elastic(imgs)
        # Batch-level stochastic intensity ops
        if random.random() < 0.40:
            imgs = torch.clamp(imgs + torch.randn_like(imgs) * 0.03, 0.0, 1.0)
        if random.random() < 0.35:
            imgs = torch.clamp(imgs * (0.75 + 0.5 * torch.rand(imgs.size(0), 1, 1, 1, device=imgs.device)), 0, 1)
        return imgs
    
    # FIX: label_smoothing 0.05→0.02; heavy smoothing softens adversarial gradients
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    ema = ModelEMA(model, decay=0.995)
    scaler = GradScaler('cuda', enabled=USE_CUDA)  # AMP: ~2x throughput on RX 6600 FP16
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dev_name = torch.cuda.get_device_name(DEVICE) if USE_CUDA else "CPU"
    print("\n" + "=" * 65)
    print("           CAPTCHA TEACHER CRNN TRAINING ENGINE")
    print("=" * 65)
    print(f"[*] Compute Device     : {DEVICE} ({dev_name})")
    print(f"[*] Train / Test Split : {len(train_ds):,} / {len(test_ds):,} samples")
    print(f"[*] Trainable Params   : {total_params:,}")
    print(f"[*] Batch Size / Epoch : {BATCH_SIZE} / {EPOCHS}")
    print(f"[*] Initial LR         : {LR}")
    print("=" * 65 + "\n")

    best_acc = 0.0
    no_improve_epochs = 0

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        train_hard_indices = [] # EXPERT FIX: Track hard samples DURING training!
        
        epoch_start_time = time.time()
        for batch_idx, (imgs, tgts, _, idxs) in enumerate(train_loader):
            imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
            imgs = augmenter_fn(imgs, epoch)  # Kornia GPU augmentation

            # 1. ZERO GRADIENTS AT THE START OF EACH BATCH
            optimizer.zero_grad()

            # --- PHASE 1: GENERATE ADVERSARIAL ATTACK (PGD) ---
            temp = 0.7 + 0.3 * np.exp(-epoch / 40.0)
            # Randomized epsilon ceiling → model sees varied attack intensities
            epsilon_ceil = random.uniform(0.04, 0.10)
            epsilon = epsilon_ceil * (1 - np.exp(-epoch / 10))
            alpha = epsilon / 4

            attack_model = ema.ema_model if random.random() < 0.7 else model
            #attack_model.eval()
            # 2 restarts, fewer iters: same compute, explores 2 starting points
            # → keeps worst-case perturbation → stronger robustness signal
            # iters = min(5, 1 + epoch // 4) if USE_CUDA else 1
            iters = 2 if USE_CUDA else 1
            perturbed_imgs = pgd_attack(attack_model, imgs, tgts, epsilon, alpha,
                                        iters, criterion, temp, n_restarts=2)
            model.train()

            # --- PHASE 2: STANDARD FORWARD PASS (AMP autocast) ---
            with autocast(device_type='cuda', enabled=USE_CUDA):
                # Clean Data Pass
                out, attn_weights = model(imgs, temperature=temp)
                loss_ce = criterion(out.view(-1, NUM_CLASSES), tgts.view(-1))

                # Diversity Loss — clamped to prevent destabilizing attention early
                attn_t = attn_weights.transpose(1, 2)
                sim = torch.bmm(attn_t, attn_weights)
                seq_len = attn_weights.size(2)
                identity = torch.eye(seq_len, device=sim.device).unsqueeze(0)
                div_loss = (sim * (1 - identity)).mean().clamp(max=1.0)

                total_loss = loss_ce + (0.003 * div_loss)

                # Poisoned Data Pass
                out_adv, _ = model(perturbed_imgs, temperature=temp)
                loss_adv = criterion(out_adv.view(-1, NUM_CLASSES), tgts.view(-1))

                # Balanced combination: keeps total loss scale consistent.
                # Previous: total_loss + adv_weight*adv doubled CE weight for clean pass.
                adv_weight = min(0.5, max(0.0, (epoch - 3) / 10))
                combined_loss = (1.0 - adv_weight) * total_loss + adv_weight * loss_adv

            # --- PHASE 3: BACKWARD PASS & EMA UPDATE ---
            scaler.scale(combined_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA decay: ~0.945 at start, asymptotically → 0.995 by epoch ~30
            ema.decay = 0.995 - 0.05 * np.exp(-epoch / 10)
            ema.update(model)

            t_loss += combined_loss.item()

            # Confidence-based hard mining: wrong predictions AND low-confidence
            # correct ones are both hard — a barely-right answer is still risky.
            with torch.no_grad():
                preds = out.argmax(dim=-1)
                is_correct = (preds == tgts).all(dim=1).cpu()
                probs_train = torch.softmax(out.float(), dim=-1)
                conf_train = probs_train.max(dim=-1).values.mean(dim=1).cpu()
            # Adaptive confidence threshold: permissive early, strict late.
            # Avoids flagging everything as hard in epoch 0 (when conf is always low).
            conf_thresh = 0.60 + 0.25 * (1.0 - np.exp(-epoch / 10))
            is_hard_mask = ~is_correct | (conf_train < conf_thresh)
            train_hard_indices.extend(idxs[is_hard_mask].tolist())
            # Live inline step progress
            current_step = batch_idx + 1
            total_steps = len(train_loader)
            sys.stdout.write(
                f"\r--> Epoch [{epoch+1:02d}/{EPOCHS:02d}] "
                f"Step [{current_step:03d}/{total_steps:03d}] "
                f"| Loss: {combined_loss.item():.4f} "
                f"| Adv: {loss_adv.item():.4f} "
            )
            sys.stdout.flush()
                        
        sys.stdout.write("\n")
        # --- TRUE DYNAMIC HARD SAMPLE MINING ---
        # Hard decay: 0.95 (not 0.98) — we re-mark each epoch, so old hardness
        # should clear faster. 0.98 ≈ half-life 34 epochs; 0.95 ≈ 13 epochs.
        full_dataset.is_hard *= 0.95
        if len(train_hard_indices) > 0:
            full_dataset.is_hard[train_hard_indices] = 1
        # FIX: continuous smooth weighting using the decayed is_hard float value.
        # Binary 3.0/1.0 threw away the decay signal we carefully computed.
        # fresh hard → weight ~3.0, fading hard → smoothly returns to 1.0
        new_weights = [1.0 + 2.0 * float(full_dataset.is_hard[idx]) for idx in train_ds.indices]
        sampler.weights = torch.as_tensor(new_weights, dtype=torch.double)
            
        # --- EVALUATION PHASE ---
        model.eval()
        correct_words = 0
        char_correct_total = 0
        total_chars = 0
        val_loss_total = 0.0
        current_hard_samples = set()

        target_accuracy = np.clip(TARGET_ACCURACY, 0.0, 100.0)

        print(f"\nEpoch {epoch+1} Tests:")
        # Threshold: starts permissive, tightens as model matures
        threshold = 0.90 - 0.30 * np.exp(-epoch / 8)
        with torch.no_grad():
            for batch_idx, (imgs, tgts, pseudo_filenames, _) in enumerate(test_loader):
                imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)

                out, _ = ema.ema_model(imgs)
                val_loss = criterion(out.view(-1, NUM_CLASSES), tgts.view(-1))
                val_loss_total += val_loss.item()

                preds = out.argmax(dim=-1)

                # Accuracy metrics (Word-level and Char-level separate)
                is_correct = (preds == tgts).all(dim=1)
                correct_words += is_correct.sum().item()

                char_correct_total += (preds == tgts).sum().item()
                total_chars += preds.numel()

                probs = torch.softmax(out, dim=-1)
                conf = probs.max(dim=-1).values.mean(dim=1)

                for i in range(len(imgs)):
                    if not is_correct[i] or conf[i].item() < threshold:
                        current_hard_samples.add(pseudo_filenames[i])

                if batch_idx == 0:
                    for i in range(min(5, len(preds))):
                        p_str = "".join([IDX2CHAR.get(c.item(), '?') for c in preds[i]])
                        t_str = "".join([IDX2CHAR.get(c.item(), '?') for c in tgts[i]])
                        print(f"   Prediction: {p_str.ljust(6)} | Word: {t_str}")

        word_acc = (correct_words / test_size) * 100
        char_acc = (char_correct_total / total_chars) * 100
        val_loss_avg = val_loss_total / len(test_loader)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time

        print(f"\n[+] Epoch {epoch+1:02d} Summary ({epoch_time:.2f}s):")
        print(f"    Train Loss : {t_loss/len(train_loader):.4f} | Val Loss : {val_loss_avg:.4f}")
        print(f"    Word Acc   : {word_acc:.2f}% | Char Acc : {char_acc:.2f}% | LR: {current_lr:.6f}")

        # Scheduler step (Char accuracy driven)
        scheduler.step(char_acc)

        if word_acc > best_acc:
            print(f"    >>> [★] NEW BEST MODEL RECORD! ({word_acc:.2f}% > {best_acc:.2f}%) Saved.")
            best_acc = word_acc
            torch.save(ema.ema_model.state_dict(), TEACHER_MODEL_PATH)
            no_improve_epochs = 0
            with open(HARD_SAMPLES_LOG, "w") as f:
                for fname in current_hard_samples:
                    f.write(f"{fname}\n")
        else:
            no_improve_epochs += 1
            print(f"    [-] Patience: {no_improve_epochs}/{NO_IMPROVEMENT_PATIENCE} (Best: {best_acc:.2f}%)")

        print("-" * 65)

        if no_improve_epochs >= NO_IMPROVEMENT_PATIENCE:
            print(f"\n[!] Early stopping triggered. No validation improvement for {NO_IMPROVEMENT_PATIENCE} epochs.")
            break
        elif word_acc >= target_accuracy:
            print(f"\n[+] Target accuracy reached ({target_accuracy}%). Training complete!")
            break

    print(f"\n[*] Training finalized. Peak Word Accuracy: {best_acc:.2f}%\n")

if __name__ == "__main__":
    print("Starting training...")
    train()
