import os
import torch
import torch.nn as nn
import numpy as np
from config import *
from models import CaptchaAttentionCRNN, CaptchaStudentCNN
from PIL import Image

# --- SETTINGS ---
EPSILON = 0.05  # Maximum allowed perturbation
ALPHA = 0.01    # Step size for each PGD iteration
ITERS = 10      # Number of PGD iterations (The higher, the stronger the attack)

def save_adversarial_png(orig_tensor, adv_tensor, init_pred, final_pred, index, model_name):
    """
    Combines the original and poisoned images side-by-side and saves as PNG.
    """
    os.makedirs(os.path.join(FILES_DIR, "attacks"), exist_ok=True)

    def tensor_to_pil(tensor):
        img = tensor.squeeze().cpu().detach().numpy()
        img_uint8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        inverted_img = 255 - img_uint8
        pil_img = Image.fromarray(inverted_img, mode='L')
        return pil_img.resize((IMG_WIDTH * 4, IMG_HEIGHT * 4), Image.Resampling.NEAREST)

    pil_orig = tensor_to_pil(orig_tensor)
    pil_adv = tensor_to_pil(adv_tensor)

    total_width = pil_orig.width + pil_adv.width + 10
    combined = Image.new('L', (total_width, pil_orig.height), 255)
    
    combined.paste(pil_orig, (0, 0))
    combined.paste(pil_adv, (pil_orig.width + 10, 0))

    filename = f"idx{index}_{model_name}_ORIG_{init_pred}_ADV_{final_pred}.png"
    filepath = os.path.join(FILES_DIR, "attacks", filename)
    
    combined.save(filepath)
    return filepath

def pgd_attack_eval(model, image, target_tensor, epsilon, alpha, iters, is_student=False):
    """
    Iterative Projected Gradient Descent (PGD) attack.
    Customized to handle differences between Teacher and Student architectures.
    """
    criterion = nn.CrossEntropyLoss()
    
    # Clone the image and add initial random noise (helps escape local minima)
    perturbed_image = image.clone().detach()
    perturbed_image = perturbed_image + torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    for _ in range(iters):
        perturbed_image.requires_grad = True
        
        # Forward pass (Handle architecture differences)
        if is_student:
            out = model(perturbed_image)
        else:
            out, _ = model(perturbed_image, temperature=1.0, return_attention=True)
            
        loss = criterion(out.view(-1, NUM_CLASSES), target_tensor.view(-1))
        
        # Calculate gradients
        model.zero_grad()
        loss.backward()
        
        data_grad = perturbed_image.grad.data
        
        # Update perturbed image iteratively
        with torch.no_grad():
            perturbed_image = perturbed_image + alpha * data_grad.sign()
            # Projection step: ensure perturbation stays within [-epsilon, epsilon] bounds
            eta = torch.clamp(perturbed_image - image, min=-epsilon, max=epsilon)
            # Ensure pixel values stay valid [0, 1]
            perturbed_image = torch.clamp(image + eta, min=0, max=1)
            
    return perturbed_image.detach()

def decode_prediction(preds):
    return "".join([IDX2CHAR.get(c.item(), '?') for c in preds[0]])

def benchmark_models():
    print(f"[!] Loading Teacher Model: {TEACHER_MODEL_PATH}")
    teacher = CaptchaAttentionCRNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE))
    teacher.eval() 

    print(f"[!] Loading Student Model: {STUDENT_MODEL_PATH}")
    student = CaptchaStudentCNN(NUM_CLASSES, num_characters=CAPTCHA_LENGTH).to(DEVICE)
    student.load_state_dict(torch.load(STUDENT_MODEL_PATH, map_location=DEVICE))
    student.eval()

    dataset_path = GOLDEN_DATASET if os.path.exists(GOLDEN_DATASET) else MASTER_DATASET
    data = np.load(dataset_path)
    images, labels = data['images'], data['labels']
    
    teacher_broken = 0
    student_broken = 0
    valid_attempts = 0
    max_eval_samples = 150 # Toplam kaç resim tarayacağız
    
    print(f"\n[!] VS Benchmark Starting: Teacher vs Student")
    print(f"[!] Attack Specs - PGD (Eps: {EPSILON}, Alpha: {ALPHA}, Iters: {ITERS})")
    print("-" * 65)

    for i in range(len(images)):
        if valid_attempts >= max_eval_samples:
            break

        img_arr = images[i]
        if img_arr.ndim == 2:
            img_tensor = torch.FloatTensor(img_arr).unsqueeze(0).unsqueeze(0)
        elif img_arr.ndim == 3:
            img_tensor = torch.FloatTensor(img_arr).unsqueeze(0)
        else:
            img_tensor = torch.FloatTensor(img_arr)
        img_tensor = img_tensor.to(DEVICE)
        target_str = str(labels[i]).strip()
        
        # Sadece geçerli labelları test et
        if len(target_str) != CAPTCHA_LENGTH or not all(c in CHAR2IDX for c in target_str):
            continue
            
        target_tensor = torch.LongTensor([[CHAR2IDX[c] for c in target_str]]).to(DEVICE)
        
        # 1. Normal Forward Pass (Her iki model de başlangıçta temiz resimde doğru bilmeli)
        with torch.no_grad():
            out_t, _ = teacher(img_tensor, temperature=1.0, return_attention=True)
            pred_t_str = decode_prediction(out_t.argmax(dim=-1))
            
            out_s = student(img_tensor)
            pred_s_str = decode_prediction(out_s.argmax(dim=-1))
            
        # Adil bir kıyaslama için, eğer ikisinden biri bile temiz resimde hata yaparsa pas geç
        if pred_t_str != target_str or pred_s_str != target_str:
            continue
            
        valid_attempts += 1
            
        # 2. Attack TEACHER (Teacher'ın kendi zayıf noktasına göre gradient al)
        adv_teacher = pgd_attack_eval(teacher, img_tensor, target_tensor, EPSILON, ALPHA, ITERS, is_student=False)
        with torch.no_grad():
            out_adv_t, _ = teacher(adv_teacher, temperature=1.0, return_attention=True)
            final_t_str = decode_prediction(out_adv_t.argmax(dim=-1))
            
        if final_t_str != target_str:
            teacher_broken += 1
            if teacher_broken <= 10: # Sadece ilk 10 kırılmayı kaydet (klasör dolmasın)
                save_adversarial_png(img_tensor, adv_teacher, pred_t_str, final_t_str, i, "TEACHER")

        # 3. Attack STUDENT (Student'ın kendi zayıf noktasına göre gradient al)
        adv_student = pgd_attack_eval(student, img_tensor, target_tensor, EPSILON, ALPHA, ITERS, is_student=True)
        with torch.no_grad():
            out_adv_s = student(adv_student)
            final_s_str = decode_prediction(out_adv_s.argmax(dim=-1))
            
        if final_s_str != target_str:
            student_broken += 1
            if student_broken <= 10: # Sadece ilk 10 kırılmayı kaydet
                save_adversarial_png(img_tensor, adv_student, pred_s_str, final_s_str, i, "STUDENT")

        # Konsola canlı rapor
        print(f"Sample {i:04d} | Target: {target_str} | Teacher: {'BROKEN' if final_t_str != target_str else 'SAFE'} | Student: {'BROKEN' if final_s_str != target_str else 'SAFE'}")

    print("-" * 65)
    print("=== ROBUSTNESS BENCHMARK REPORT ===")
    print(f"Total Valid Samples Tested: {valid_attempts} (Images both models initially predicted correctly)")
    print(f"Teacher Model Broken : {teacher_broken} times ({(teacher_broken/valid_attempts)*100:.1f}%)")
    print(f"Student Model Broken : {student_broken} times ({(student_broken/valid_attempts)*100:.1f}%)")
    
    if student_broken < teacher_broken:
        print("\nCONCLUSION: The Student is MORE ROBUST than the Teacher!")
    elif student_broken == teacher_broken:
        print("\nCONCLUSION: Both models show equal robustness.")
    else:
        print("\nCONCLUSION: The Teacher is still slightly more robust than the Student.")

if __name__ == "__main__":
    benchmark_models()