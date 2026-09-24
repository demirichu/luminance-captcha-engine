import os
import torch
import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, CalibrationMethod
from onnxsim import simplify
from config import NUM_CLASSES, CAPTCHA_LENGTH, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, STUDENT_ONNX_PATH, STUDENT_INT8_PATH, FINAL_DEPLOY_MODEL, GOLDEN_DATASET, MASTER_DATASET, STUDENT_MODEL_PATH, TEACHER_MODEL_PATH, TEACHER_ONNX_PATH, DEVICE
from models import CaptchaAttentionCRNN, CaptchaStudentCNN

# --- 1. CALIBRATION DATA READER ---
class CaptchaDataReader(CalibrationDataReader):
    def __init__(self, dataset_path, batch_size=1):
        data = np.load(dataset_path)
        # For the armored model, Entropy calibration with 1000 samples is best
        self.images = data['images'][:1000].astype(np.float32)
        if self.images.ndim == 3:
            self.images = np.expand_dims(self.images, axis=1) # (1000, 1, H, W)
        self.enum_data = iter([{'input': self.images[i:i+batch_size]} for i in range(len(self.images))])

    def get_next(self):
        return next(self.enum_data, None)

def kill_and_clean():
    fp32_path = STUDENT_ONNX_PATH
    int8_path = STUDENT_INT8_PATH
    final_path = FINAL_DEPLOY_MODEL
    dataset_path = GOLDEN_DATASET if os.path.exists(GOLDEN_DATASET) else MASTER_DATASET

    print(f"--- PHASE 1: INT8 Static Quantization (Entropy) ---")
    dr = CaptchaDataReader(dataset_path)

    quantize_static(
        fp32_path,
        int8_path,
        dr,
        quant_format=onnxruntime.quantization.QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Entropy,
        # --- WELDING GUARANTEE (SINGLE FILE) ---
        use_external_data_format=False, 
        extra_options={
            'WeightSymmetric': True,
            'ActivationSymmetric': False
        }
    )
    print("[!] Quantization Completed.")

    print(f"\n--- PHASE 2: ONNX Simplification (Cleaning) ---")
    # Load the quantized model
    model = onnx.load(int8_path)
    
    # Run the simplifier
    model_simp, check = simplify(model)

    if check:
        # Instruct ONNX to save as a single file without leaking data externally
        onnx.save(model_simp, final_path)
        print(f"[!] Model cleaned, paths stripped, and sealed as a SINGLE FILE: {final_path}")
    else:
        print("[X] Simplification failed, keeping original INT8.")
        onnx.save(model, final_path)

    # --- REALISTIC OPERATION REPORT ---
    # If FP32 has a hidden .data file, account for it (e.g., 50KB + 1143KB)
    fp32_size_bytes = os.path.getsize(fp32_path)
    if os.path.exists(fp32_path + ".data"):
        fp32_size_bytes += os.path.getsize(fp32_path + ".data")
        
    fp32_size = fp32_size_bytes / 1024
    final_size = os.path.getsize(final_path) / 1024
    
    print(f"\n--- OPERATION REPORT ---")
    print(f"Original FP32 (Skeleton + Brain): {fp32_size:.2f} KB")
    print(f"Final INT8 (Welded Single Piece): {final_size:.2f} KB")
    print(f"True Compression Ratio: {((1 - final_size/fp32_size)*100):.1f}%")

    print(f"\n--- PHASE 3: Residue Cleanup ---")
    try:
        if os.path.exists(fp32_path): os.remove(fp32_path)
        if os.path.exists(fp32_path + ".data"): os.remove(fp32_path + ".data")
        if os.path.exists(int8_path): os.remove(int8_path)
        print("[!] Intermediate ONNX files deleted successfully.")
        print(f"[!] FINAL ARMORED MODEL: {final_path}")
    except Exception as e:
        print(f"[X] Could not delete some residues: {e}")

def export_teacher_to_onnx():
    model = CaptchaAttentionCRNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)
    if not os.path.exists(TEACHER_MODEL_PATH): return
    model.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    dummy_input = torch.randn(1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH).to(DEVICE)
    
    # ONNX Export fix: specifying 2 outputs (out and attn_weights)
    torch.onnx.export(
        model, dummy_input, TEACHER_ONNX_PATH, 
        export_params=True, opset_version=18, do_constant_folding=True, 
        input_names=['input'], output_names=['output', 'attention_weights'],
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'},
            'attention_weights': {0: 'batch_size'}
        }
    )
    print(f"\nONNX Export Successful: {TEACHER_ONNX_PATH}")

def export_student_to_onnx():
    print("\nConverting Student model to ONNX format...")
    student = CaptchaStudentCNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to('cpu')
    student.load_state_dict(torch.load(STUDENT_MODEL_PATH, map_location='cpu'))
    student.eval()
    
    dummy_input = torch.randn(1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH).to('cpu')
    
    torch.onnx.export(
        student, dummy_input, STUDENT_ONNX_PATH, 
        export_params=True, 
        opset_version=18, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], # Only one output
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )
    print(f"FP32 ONNX Export Successful: {STUDENT_ONNX_PATH}")

if __name__ == "__main__":
    export_student_to_onnx()
    kill_and_clean()