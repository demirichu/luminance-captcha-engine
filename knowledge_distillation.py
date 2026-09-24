import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import kornia.augmentation as K
import random

from config import *
from models import CaptchaAttentionCRNN, CaptchaStudentCNN

# --- DATASET ---
class DistillDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self): return len(self.images)

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
        targets = [CHAR2IDX[c] for c in label_str]  # pre-filtered, safe direct lookup
        return tensor, torch.LongTensor(targets)


# --- DISTILLATION LOSS ---
def distillation_loss(student_logits, teacher_logits, targets, temp=3.0, alpha=0.3):
    # Hard label loss
    hard_loss = F.cross_entropy(student_logits.view(-1, NUM_CLASSES), targets.view(-1))
    # Soft label loss (KL divergence)
    soft_targets = F.softmax(teacher_logits / temp, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    soft_loss = F.kl_div(student_log_probs, soft_targets, reduction='batchmean')
    return (alpha * hard_loss) + ((1.0 - alpha) * (temp ** 2) * soft_loss)

def _pgd_attack(model, images, labels, epsilon, alpha, iters=3):
    """Lightweight PGD for student hardening.
    Uses eval() for deterministic gradients (BN/Dropout stable).
    Full FP32 for precise gradient direction."""
    criterion = nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()  # deterministic gradients, matches train.py pgd_attack
    images = images.clone().detach()
    perturbed = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0, 1)
    for _ in range(iters):
        perturbed = perturbed.detach()  # sever graph each iter
        perturbed.requires_grad_(True)
        out = model(perturbed)
        loss = criterion(out.view(-1, NUM_CLASSES), labels.view(-1))
        grad = torch.autograd.grad(loss, perturbed, only_inputs=True)[0]
        with torch.no_grad():
            perturbed = perturbed + alpha * grad.sign()
            eta = torch.clamp(perturbed - images, -epsilon, epsilon)
            perturbed = torch.clamp(images + eta, 0, 1)
    if was_training:
        model.train()  # restore original mode
    return perturbed.detach()

def train_distillation():
    # Pre-filter bad labels once at load time (mirrors train.py approach)
    dataset_path = GOLDEN_DATASET if os.path.exists(GOLDEN_DATASET) else MASTER_DATASET
    _raw = np.load(dataset_path)
    _raw_imgs, _raw_labels = _raw['images'], _raw['labels']
    _valid = np.array([
        len(str(l).strip()) == CAPTCHA_LENGTH and all(c in CHAR2IDX for c in str(l).strip())
        for l in _raw_labels
    ])
    if _valid.sum() < len(_raw_labels):
        print(f"[Distill] Filtered {(~_valid).sum()} invalid labels.")
    dataset = DistillDataset(_raw_imgs[_valid], _raw_labels[_valid])
    train_size = int(0.9 * len(dataset))
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=4 if torch.cuda.is_available() else 0,
                              pin_memory=torch.cuda.is_available(),
                              persistent_workers=True if torch.cuda.is_available() else False,
                              prefetch_factor=2 if torch.cuda.is_available() else None)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False,
                              num_workers=4 if torch.cuda.is_available() else 0,
                              pin_memory=torch.cuda.is_available(),
                              persistent_workers=True if torch.cuda.is_available() else False,
                              prefetch_factor=2 if torch.cuda.is_available() else None)

    # --- GPU-NATIVE AUGMENTATION (Kornia, matches train.py pipeline) ---
    # Teacher always sees clean data; student sees augmented clean + adversarial.
    _distill_aug = nn.Sequential(
        K.RandomAffine(degrees=5, translate=(0.04, 0.04), p=0.8),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.2), p=0.30),
        K.RandomErasing(scale=(0.01, 0.05), ratio=(0.5, 2.0), value=0.0, p=0.20), # Student'a captcha silinme efekti
    ).to(DEVICE)

    def _distill_augment(imgs):
        imgs = _distill_aug(imgs)
        if random.random() < 0.40:
            imgs = torch.clamp(imgs + torch.randn_like(imgs) * 0.025, 0.0, 1.0)
        return imgs

    print(f"Loading Teacher: {TEACHER_MODEL_PATH}")
    teacher = CaptchaAttentionCRNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE))
    teacher.eval() # CRITICAL: Teacher is never trained, only provides wise answers.
    
    student = CaptchaStudentCNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = GradScaler('cuda', enabled=USE_CUDA)  # AMP for student forward passes
    
    best_acc = 0.0
    EPOCHS = 40

    no_improve_epochs = 0
    target_accuracy = np.clip(TARGET_ACCURACY, 0.0, 100.0)
    
    for epoch in range(EPOCHS):
        student.train()
        t_loss = 0
        # Soft→Hard curriculum: wide soft labels early, sharper hard labels late
        temp  = max(1.5, 4.0 - epoch * 0.065)   # 4.0 → ~1.5 over 40 epochs
        alpha = min(0.50, 0.20 + epoch * 0.008)  # 0.20 → 0.50 (hard label weight)
        
        for imgs, tgts in train_loader:
            imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
            
            # Teacher on clean data (frozen reference, no grad needed)
            with torch.no_grad():
                teacher_logits, _ = teacher(imgs, return_attention=True)

            # PGD: student hardened against its own decision boundary
            eps = random.uniform(0.02, 0.06)
            perturbed_imgs = _pgd_attack(student, imgs, tgts, eps, eps / 4, iters=3)

            optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=USE_CUDA):
                # Clean pass: augmented student input vs teacher soft labels
                imgs_aug = _distill_augment(imgs)
                student_logits_clean = student(imgs_aug)
                loss_clean = distillation_loss(student_logits_clean, teacher_logits,
                                               tgts, temp=temp, alpha=alpha)

                # Adversarial pass: poisoned input must still match teacher's clean answer
                student_adv_logits = student(perturbed_imgs)
                loss_adv = distillation_loss(student_adv_logits, teacher_logits,
                                             tgts, temp=temp, alpha=alpha)

                # Balanced: same loss-scale convention as train.py
                final_loss = 0.5 * loss_clean + 0.5 * loss_adv

            scaler.scale(final_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss += final_loss.item()
            
        # --- VALIDATION ---
        student.eval()
        correct_words = 0
        char_correct = 0
        total_chars = 0
        with torch.no_grad():
            for imgs_v, tgts_v in test_loader:
                imgs_v, tgts_v = imgs_v.to(DEVICE), tgts_v.to(DEVICE)
                out_v = student(imgs_v)
                preds = out_v.argmax(dim=-1)
                is_correct = (preds == tgts_v).all(dim=1)
                correct_words += is_correct.sum().item()
                char_correct += (preds == tgts_v).sum().item()
                total_chars += preds.numel()

        acc = (correct_words / len(test_ds)) * 100
        char_acc = (char_correct / total_chars) * 100
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {t_loss/len(train_loader):.4f} | "
              f"LR: {current_lr:.5f} | Word: %{acc:.2f} | Char: %{char_acc:.2f}")

        scheduler.step(char_acc)  # char_acc smoother signal, matches train.py convention

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), STUDENT_MODEL_PATH)
            no_improve_epochs = 0
            print(f"--> New Record! Armored Student Saved: %{acc:.2f}")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= NO_IMPROVEMENT_PATIENCE or acc >= target_accuracy:
            print(f"Early stopping: No improvement for {NO_IMPROVEMENT_PATIENCE} epochs.")
            break

if __name__ == "__main__":
    train_distillation()