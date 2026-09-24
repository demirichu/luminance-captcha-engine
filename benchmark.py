"""
CAPTCHA AI Pipeline - Comprehensive Benchmark & Evaluation Utility

Evaluates Clean Accuracy, Character-Level Accuracy, Model Confidence,
Adversarial PGD Robustness, Single-Sample Latency, and Model Footprint.

Supports:
  - Full comparative suite across all architectures (Teacher, Student, Quantized INT8)
  - Single model evaluation (--model)
  - Single image inference (--image)
"""

import os
import sys
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import onnxruntime as ort

from config import (
    DEVICE,
    CHAR2IDX,
    IDX2CHAR,
    NUM_CLASSES,
    CAPTCHA_LENGTH,
    IMG_WIDTH,
    IMG_HEIGHT,
    IMG_CHANNELS,
    GOLDEN_DATASET,
    MASTER_DATASET,
    TEACHER_MODEL_PATH,
    STUDENT_MODEL_PATH,
    FINAL_DEPLOY_MODEL
)
from models import CaptchaAttentionCRNN, CaptchaStudentCNN

# Default PGD Robustness Hyperparameters
DEFAULT_PGD_EPSILON = 0.03
DEFAULT_PGD_ALPHA = 0.008
DEFAULT_PGD_ITERS = 5
DEFAULT_PGD_SAMPLES = 1000


def resolve_existing_path(configured_path: str) -> str:
    """Resolves file path checking configured path, root directory, or files/."""
    if not configured_path:
        return None
    if os.path.exists(configured_path):
        return configured_path

    base_name = os.path.basename(configured_path)
    if os.path.exists(base_name):
        return base_name

    sub_path = os.path.join("files", base_name)
    if os.path.exists(sub_path):
        return sub_path

    return None


def softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for NumPy arrays."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def load_dataset(npz_path: str, max_samples: int = None):
    """Loads and sanitizes dataset arrays regardless of numpy archive key naming."""
    print(f"[*] Reading dataset: {npz_path}")
    data = np.load(npz_path)

    img_data = None
    lbl_data = None

    if 'images' in data:
        img_data = data['images']
    if 'labels' in data:
        lbl_data = data['labels']

    if img_data is None or lbl_data is None:
        for key in data.files:
            arr = data[key]
            if img_data is None and arr.ndim >= 2 and arr.shape[-1] == IMG_WIDTH:
                img_data = arr
            elif lbl_data is None and arr.ndim == 1:
                lbl_data = arr

    if img_data is None and 'arr_0' in data:
        img_data = data['arr_0']
    if lbl_data is None and 'arr_1' in data:
        lbl_data = data['arr_1']

    if img_data is None or lbl_data is None:
        raise ValueError(f"Could not parse valid image/label arrays. Found keys: {data.files}")

    images = img_data.astype(np.float32)
    if images.max() > 1.0:
        images = images / 255.0
    if images.ndim == 3:
        images = np.expand_dims(images, axis=1)

    valid_images = []
    numeric_labels = []

    for idx, raw_lbl in enumerate(lbl_data):
        lbl_str = str(raw_lbl).strip()
        if lbl_str.startswith("b'") or lbl_str.startswith('b"'):
            lbl_str = lbl_str[2:-1]

        if len(lbl_str) == CAPTCHA_LENGTH and all(c in CHAR2IDX for c in lbl_str):
            numeric_labels.append([CHAR2IDX[c] for c in lbl_str])
            valid_images.append(images[idx])

    final_images = np.array(valid_images, dtype=np.float32)
    final_labels = np.array(numeric_labels, dtype=np.int64)

    if max_samples and 0 < max_samples < len(final_images):
        final_images = final_images[:max_samples]
        final_labels = final_labels[:max_samples]

    return final_images, final_labels


def preprocess_single_image(image_path: str) -> np.ndarray:
    """Loads and preprocesses an image file into shape (1, 1, 17, 60) in [0, 1]."""
    img = Image.open(image_path)
    if img.mode in ('RGBA', 'LA'):
        alpha = np.array(img.split()[-1], dtype=np.float32)
        rgb = np.array(img.convert('RGB'), dtype=np.float32)
        arr = alpha if (alpha.std() > 5.0 and rgb.std() < 5.0) else np.array(img.convert('L'), dtype=np.float32)
    else:
        arr = np.array(img.convert('L'), dtype=np.float32)

    if arr.shape != (IMG_HEIGHT, IMG_WIDTH):
        pil_resized = Image.fromarray(arr.astype(np.uint8)).resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
        arr = np.array(pil_resized, dtype=np.float32)

    if arr.max() > 1.0:
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)[np.newaxis, np.newaxis, :, :].astype(np.float32)


def run_pgd_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor,
                   device: torch.device, epsilon: float, alpha: float, iters: int) -> torch.Tensor:
    """Generates PGD adversarial perturbations on PyTorch models using sign gradients."""
    was_training = model.training
    model.train()

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
            m.eval()

    criterion = nn.CrossEntropyLoss()

    x = images.clone().detach().to(device)
    y = labels.to(device)

    perturbed = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
    perturbed = torch.clamp(perturbed, 0.0, 1.0)

    for _ in range(iters):
        perturbed.requires_grad_(True)
        out = model(perturbed)
        if isinstance(out, tuple):
            out = out[0]

        loss = criterion(out.view(-1, NUM_CLASSES), y.view(-1))
        grad = torch.autograd.grad(loss, perturbed, only_inputs=True)[0]

        with torch.no_grad():
            perturbed = perturbed + alpha * grad.sign()
            eta = torch.clamp(perturbed - x, min=-epsilon, max=epsilon)
            perturbed = torch.clamp(x + eta, min=0.0, max=1.0)
            perturbed = perturbed.detach()

    if not was_training:
        model.eval()

    return perturbed


def evaluate_pytorch_model(model: nn.Module, images: np.ndarray, labels: np.ndarray,
                           device: torch.device, epsilon: float = DEFAULT_PGD_EPSILON,
                           alpha: float = DEFAULT_PGD_ALPHA, iters: int = DEFAULT_PGD_ITERS,
                           max_pgd_samples: int = None, run_pgd: bool = True,
                           warmup: int = 15, latency_iters: int = 80):
    """Measures clean accuracy, character accuracy, confidence, PGD robustness, and latency."""
    model.eval()
    x_tensor = torch.from_numpy(images).to(device)
    y_tensor = torch.from_numpy(labels).to(device)

    total_samples = len(y_tensor)
    correct_clean = 0
    total_char_correct = 0
    all_confs = []
    batch_size = 128

    # 1. Clean Accuracy & Confidence Benchmark
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            bx = x_tensor[i:i + batch_size]
            by = y_tensor[i:i + batch_size]
            out = model(bx)
            if isinstance(out, tuple):
                out = out[0]

            probs = torch.softmax(out, dim=-1)
            preds = probs.argmax(dim=-1)

            correct_clean += (preds == by).all(dim=-1).sum().item()
            total_char_correct += (preds == by).sum().item()

            confs = torch.gather(probs, 2, preds.unsqueeze(-1)).squeeze(-1)
            all_confs.append(confs.mean(dim=-1).cpu().numpy())

    clean_accuracy = (correct_clean / total_samples) * 100.0
    char_accuracy = (total_char_correct / (total_samples * CAPTCHA_LENGTH)) * 100.0
    mean_confidence = float(np.mean(np.concatenate(all_confs))) * 100.0

    # 2. PGD Adversarial Robustness Benchmark
    pgd_accuracy = None
    combined_adv_samples = None

    if run_pgd:
        pgd_sample_count = min(total_samples, max_pgd_samples) if max_pgd_samples else total_samples
        pgd_x = x_tensor[:pgd_sample_count]
        pgd_y = y_tensor[:pgd_sample_count]

        adversarial_tensors = []
        correct_adv = 0
        pgd_batch_size = 64

        print(f"    [>] Running PGD robustness evaluation on {pgd_sample_count} samples...")
        for i in range(0, pgd_sample_count, pgd_batch_size):
            bx = pgd_x[i:i + pgd_batch_size]
            by = pgd_y[i:i + pgd_batch_size]
            bx_adv = run_pgd_attack(model, bx, by, device, epsilon=epsilon, alpha=alpha, iters=iters)
            adversarial_tensors.append(bx_adv)

            with torch.no_grad():
                out_adv = model(bx_adv)
                if isinstance(out_adv, tuple):
                    out_adv = out_adv[0]
                preds_adv = out_adv.argmax(dim=-1)
                correct_adv += (preds_adv == by).all(dim=-1).sum().item()

        pgd_accuracy = (correct_adv / pgd_sample_count) * 100.0
        combined_adv_samples = torch.cat(adversarial_tensors, dim=0).cpu().numpy()

    # 3. Single-Sample Inference Latency Benchmark (Batch Size = 1)
    single_input = x_tensor[:1]
    for _ in range(warmup):
        _ = model(single_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(latency_iters):
        _ = model(single_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
    latency_ms = ((time.perf_counter() - start_time) / latency_iters) * 1000.0

    return clean_accuracy, char_accuracy, mean_confidence, pgd_accuracy, latency_ms, combined_adv_samples


def evaluate_onnx_model(onnx_path: str, images: np.ndarray, labels: np.ndarray,
                        transfer_adv_images: np.ndarray = None,
                        warmup: int = 15, latency_iters: int = 80):
    """Evaluates ONNX runtime engine on CPU for clean accuracy, confidence, latency, & black-box transfer."""
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # 1. Clean Accuracy & Confidence Benchmark
    total_samples = len(labels)
    correct_clean = 0
    total_char_correct = 0
    all_confs = []
    batch_size = 128

    for i in range(0, total_samples, batch_size):
        bx = images[i:i + batch_size]
        by = labels[i:i + batch_size]
        out = session.run(None, {input_name: bx})[0]
        probs = softmax_np(out)
        preds = probs.argmax(axis=-1)

        correct_clean += (preds == by).all(axis=-1).sum()
        total_char_correct += (preds == by).sum()

        confs = np.take_along_axis(probs, preds[:, :, np.newaxis], axis=-1).squeeze(-1)
        all_confs.append(confs.mean(axis=-1))

    clean_accuracy = (correct_clean / total_samples) * 100.0
    char_accuracy = (total_char_correct / (total_samples * CAPTCHA_LENGTH)) * 100.0
    mean_confidence = float(np.mean(np.concatenate(all_confs))) * 100.0

    # 2. Black-box Transfer Attack Robustness Benchmark
    pgd_accuracy = None
    if transfer_adv_images is not None:
        adv_count = len(transfer_adv_images)
        correct_adv = 0
        by_adv = labels[:adv_count]

        for i in range(0, adv_count, batch_size):
            bx = transfer_adv_images[i:i + batch_size]
            by = by_adv[i:i + batch_size]
            out = session.run(None, {input_name: bx})[0]
            preds = out.argmax(axis=-1)
            correct_adv += (preds == by).all(axis=-1).sum()

        pgd_accuracy = (correct_adv / adv_count) * 100.0

    # 3. Single-Sample Inference Latency Benchmark (Batch Size = 1)
    single_input = images[:1]
    for _ in range(warmup):
        _ = session.run(None, {input_name: single_input})

    start_time = time.perf_counter()
    for _ in range(latency_iters):
        _ = session.run(None, {input_name: single_input})
    latency_ms = ((time.perf_counter() - start_time) / latency_iters) * 1000.0

    return clean_accuracy, char_accuracy, mean_confidence, pgd_accuracy, latency_ms


def test_single_image(model_path: str, image_path: str):
    """Infers on a single image and prints prediction, character confidences, and latency."""
    print("=" * 65)
    print(f"[*] Single Image Inference: {image_path}")
    print(f"[*] Model: {model_path}")
    print("=" * 65)

    x = preprocess_single_image(image_path)
    t0 = time.perf_counter()

    if model_path.lower().endswith(".onnx"):
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        probs = softmax_np(logits)[0]
    else:
        ckpt = torch.load(model_path, map_location=DEVICE)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt.state_dict()
        if any("pos_embedding" in k or "rnn" in k for k in state_dict.keys()):
            model = CaptchaAttentionCRNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH)
        else:
            model = CaptchaStudentCNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():
            out = model(torch.from_numpy(x).to(DEVICE))
            if isinstance(out, tuple):
                out = out[0]
            probs = torch.softmax(out, dim=-1).cpu().numpy()[0]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    preds = np.argmax(probs, axis=-1)
    pred_str = "".join(IDX2CHAR[idx] for idx in preds)
    confs = [probs[i, preds[i]] * 100.0 for i in range(CAPTCHA_LENGTH)]

    print(f"Prediction : {pred_str}")
    print(f"Confidence : {np.mean(confs):.2f}% (min: {np.min(confs):.2f}%)")
    print(f"Latency    : {latency_ms:.2f} ms")
    print("-" * 65)
    for i, (c, conf) in enumerate(zip(pred_str, confs), 1):
        print(f"  Pos #{i}: '{c}' ({conf:.2f}%)")
    print("=" * 65)


def format_size(byte_count: int) -> str:
    kb = byte_count / 1024.0
    if kb < 1024:
        return f"{kb:.1f} KB"
    return f"{(kb / 1024.0):.2f} MB"


def print_cli_table(headers, rows):
    """Prints a perfectly aligned and framed ASCII table in the terminal."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep_line = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    print("\n" + sep_line)
    header_str = "| " + " | ".join([f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))]) + " |"
    print(header_str)
    print(sep_line)

    for row in rows:
        row_str = "| " + " | ".join([f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row))]) + " |"
        print(row_str)

    print(sep_line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Captcha AI Pipeline - Comprehensive Benchmark & Testing Utility")
    parser.add_argument("--dataset", type=str, default=GOLDEN_DATASET if os.path.exists(GOLDEN_DATASET) else MASTER_DATASET, help="Path to evaluation dataset (.npz)")
    parser.add_argument("--model", type=str, default=None, help="Target specific model to benchmark (.onnx or .pth)")
    parser.add_argument("--image", type=str, default=None, help="Path to single image for instant inference")
    parser.add_argument("--samples", type=int, default=None, help="Limit sample count for fast evaluation")
    parser.add_argument("--no-pgd", action="store_true", help="Skip PGD adversarial robustness attack")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_PGD_EPSILON, help="PGD perturbation bound (L-infinity)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_PGD_ALPHA, help="PGD step size")
    parser.add_argument("--iters", type=int, default=DEFAULT_PGD_ITERS, help="Number of PGD iterations")
    parser.add_argument("--pgd-samples", type=int, default=DEFAULT_PGD_SAMPLES, help="Sample count for adversarial test")
    args = parser.parse_args()

    # Mode 1: Single image inference
    if args.image:
        model_path = resolve_existing_path(args.model) if args.model else resolve_existing_path(FINAL_DEPLOY_MODEL)
        test_single_image(model_path, args.image)
        return

    print("=" * 80)
    print("CAPTCHA AI BENCHMARK ENGINE")
    print("=" * 80)
    print(f"[*] Evaluation Device: {DEVICE}")
    print(f"[*] Config: {CAPTCHA_LENGTH} chars, {IMG_WIDTH}x{IMG_HEIGHT}, {NUM_CLASSES} classes")

    dataset_path = resolve_existing_path(args.dataset)
    if not dataset_path:
        print(f"[!] Target dataset not found: {args.dataset}")
        return

    images, labels = load_dataset(dataset_path, max_samples=args.samples)
    print(f"[+] Loaded verified evaluation set: {len(images):,} samples | Shape: {images.shape}")

    benchmark_rows = []
    student_adv_cache = None
    run_pgd = not args.no_pgd

    # Mode 2: Targeted single model benchmark
    if args.model:
        m_path = resolve_existing_path(args.model)
        if not m_path:
            print(f"[!] Model not found: {args.model}")
            return

        print(f"\n[*] Evaluating Target Model: {m_path}...")
        file_size = format_size(os.path.getsize(m_path))

        if m_path.lower().endswith(".onnx"):
            c_acc, char_acc, conf, p_acc, lat = evaluate_onnx_model(m_path, images, labels)
            pgd_str = f"{p_acc:.2f}%" if p_acc is not None else "N/A"
            benchmark_rows.append([os.path.basename(m_path), "INT8/FP32", f"{c_acc:.2f}%", f"{char_acc:.2f}%", f"{conf:.2f}%", pgd_str, file_size, f"{lat:.2f} ms"])
        else:
            ckpt = torch.load(m_path, map_location=DEVICE)
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt.state_dict()
            if any("pos_embedding" in k or "rnn" in k for k in state_dict.keys()):
                m = CaptchaAttentionCRNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)
                arch_name = "Teacher CRNN"
            else:
                m = CaptchaStudentCNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)
                arch_name = "Student CNN"
            m.load_state_dict(state_dict)

            c_acc, char_acc, conf, p_acc, lat, _ = evaluate_pytorch_model(
                m, images, labels, DEVICE,
                epsilon=args.epsilon, alpha=args.alpha, iters=args.iters,
                max_pgd_samples=args.pgd_samples, run_pgd=run_pgd
            )
            pgd_str = f"{p_acc:.2f}%" if p_acc is not None else "N/A"
            benchmark_rows.append([arch_name, "FP32", f"{c_acc:.2f}%", f"{char_acc:.2f}%", f"{conf:.2f}%", pgd_str, file_size, f"{lat:.2f} ms"])

    # Mode 3: Full Comparative Benchmark across all architectures
    else:
        # 1. Benchmark Teacher CRNN (Attention)
        teacher_path = resolve_existing_path(TEACHER_MODEL_PATH)
        if teacher_path:
            print(f"\n[*] Evaluating Teacher CRNN ({teacher_path})...")
            teacher = CaptchaAttentionCRNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)
            teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))

            c_acc, char_acc, conf, p_acc, lat, _ = evaluate_pytorch_model(
                teacher, images, labels, DEVICE,
                epsilon=args.epsilon, alpha=args.alpha, iters=args.iters,
                max_pgd_samples=args.pgd_samples, run_pgd=run_pgd
            )
            file_size = format_size(os.path.getsize(teacher_path))
            pgd_str = f"{p_acc:.2f}%" if p_acc is not None else "N/A"
            benchmark_rows.append(["Teacher CRNN (Attention)", "FP32", f"{c_acc:.2f}%", f"{char_acc:.2f}%", f"{conf:.2f}%", pgd_str, file_size, f"{lat:.2f} ms"])
            print(f"    Clean Acc: {c_acc:.2f}% | Char Acc: {char_acc:.2f}% | Conf: {conf:.2f}% | PGD Acc: {pgd_str} | Latency: {lat:.2f} ms")
        else:
            print(f"[i] Teacher model checkpoint not found at: {TEACHER_MODEL_PATH} (Skipping)")

        # 2. Benchmark Student CNN
        student_path = resolve_existing_path(STUDENT_MODEL_PATH)
        if student_path:
            print(f"\n[*] Evaluating Student CNN ({student_path})...")
            student = CaptchaStudentCNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)
            student.load_state_dict(torch.load(student_path, map_location=DEVICE))

            c_acc, char_acc, conf, p_acc, lat, student_adv_cache = evaluate_pytorch_model(
                student, images, labels, DEVICE,
                epsilon=args.epsilon, alpha=args.alpha, iters=args.iters,
                max_pgd_samples=args.pgd_samples, run_pgd=run_pgd
            )
            file_size = format_size(os.path.getsize(student_path))
            pgd_str = f"{p_acc:.2f}%" if p_acc is not None else "N/A"
            benchmark_rows.append(["Student CNN", "FP32", f"{c_acc:.2f}%", f"{char_acc:.2f}%", f"{conf:.2f}%", pgd_str, file_size, f"{lat:.2f} ms"])
            print(f"    Clean Acc: {c_acc:.2f}% | Char Acc: {char_acc:.2f}% | Conf: {conf:.2f}% | PGD Acc: {pgd_str} | Latency: {lat:.2f} ms")
        else:
            print(f"[i] Student model checkpoint not found at: {STUDENT_MODEL_PATH} (Skipping)")

        # 3. Benchmark Final Deployment Engine (ONNX)
        onnx_path = resolve_existing_path(FINAL_DEPLOY_MODEL)
        if onnx_path:
            print(f"\n[*] Evaluating Final Deployment Engine ({onnx_path})...")
            c_acc, char_acc, conf, p_acc, lat = evaluate_onnx_model(onnx_path, images, labels, transfer_adv_images=student_adv_cache)
            file_size = format_size(os.path.getsize(onnx_path))
            pgd_str = f"{p_acc:.2f}%" if p_acc is not None else "N/A"
            benchmark_rows.append(["Student CNN (Quantized)", "INT8", f"{c_acc:.2f}%", f"{char_acc:.2f}%", f"{conf:.2f}%", pgd_str, file_size, f"{lat:.2f} ms"])
            print(f"    Clean Acc: {c_acc:.2f}% | Char Acc: {char_acc:.2f}% | Conf: {conf:.2f}% | PGD (Transfer): {pgd_str} | Latency: {lat:.2f} ms")
        else:
            print(f"[i] Final deployment model not found at: {FINAL_DEPLOY_MODEL} (Skipping)")

    # Render Clean Terminal CLI Table
    table_headers = [
        "Model Architecture",
        "Precision",
        "Accuracy (Clean)",
        "Char Accuracy",
        "Confidence",
        "Accuracy (PGD)",
        "Model Size",
        "Inference Latency"
    ]
    print_cli_table(table_headers, benchmark_rows)


if __name__ == "__main__":
    main()
