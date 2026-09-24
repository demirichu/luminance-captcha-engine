import os
import torch

# ============================================================
# 1. ACTIVE CAPTCHA PROFILE
# ============================================================
# Configure the parameters below for your target L-channel / grayscale CAPTCHA:
#   1. Set CAPTCHA_NAME to your identifier
#   2. Set IMG_WIDTH and IMG_HEIGHT (Height must be >= 16px)
#   3. Set CAPTCHA_LENGTH (fixed character count)
#   4. Set CHARS to your full alphabet/charset
# ============================================================

CAPTCHA_NAME = "generic_5char"
IMG_WIDTH = 120
IMG_HEIGHT = 40
IMG_CHANNELS = 1          # 1 for Grayscale / Luminance (L-channel)
CAPTCHA_LENGTH = 5
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Derived Character Mappings
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

# ============================================================
# 2. Training Hyperparameters & Compute Settings
# ============================================================

# Device Configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# Training Configuration
TARGET_ACCURACY = 100.0
FINE_TUNE_MODE = False

NO_IMPROVEMENT_PATIENCE = 10 if USE_CUDA else 7
EPOCHS = 60 if USE_CUDA else 40
BATCH_SIZE = 512 if USE_CUDA else 64
LR = (0.00008 if FINE_TUNE_MODE else 0.0005) if USE_CUDA else (0.00005 if FINE_TUNE_MODE else 0.0002)

# ============================================================
# 3. Standardized File & Artifact Paths
# ============================================================
FILES_DIR = 'files'

# Dataset Paths
RAW_DATA_DIR = os.path.join(FILES_DIR, 'raw_data')
MASTER_DATASET = os.path.join(FILES_DIR, 'master_dataset.npz')
GOLDEN_DATASET = os.path.join(FILES_DIR, 'golden_dataset.npz')
NEW_GOLDEN_DATASET = os.path.join(FILES_DIR, 'new_golden_dataset.npz')
RISKY_DATASET = os.path.join(FILES_DIR, 'risky_dataset.npz')
QUARANTINE_DATASET = os.path.join(FILES_DIR, 'quarantine_dataset.npz')
HARD_DATASET = os.path.join(FILES_DIR, 'hard_dataset.npz')
HARD_SAMPLES_LOG = os.path.join(FILES_DIR, 'hard_samples.txt')

# Model Artifacts
TEACHER_MODEL_PATH = os.path.join(FILES_DIR, 'teacher.pth')
TEACHER_ONNX_PATH = os.path.join(FILES_DIR, 'teacher.onnx')
STUDENT_MODEL_PATH = os.path.join(FILES_DIR, 'student.pth')
STUDENT_ONNX_PATH = os.path.join(FILES_DIR, 'student.onnx')
STUDENT_INT8_PATH = os.path.join(FILES_DIR, 'student_int8.onnx')
FINAL_DEPLOY_MODEL = 'captcha.onnx'
