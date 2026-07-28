import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_ROOT = 'dataset/'
SPLITS = ['train', 'val', 'test']
MIN_OBJECT_AREA = 20  # Порог в пикселях для поиска шума разметки
TARGET_CLASSES = [1, 2]  # Инициализация целевых классов объектов

report = {split: {'mismatch_size': [], 'noisy_masks': [], 'classes_per_file': {}} for split in SPLITS}

print("🔎 Запуск глубокого аудита геометрии и контуров...")

for split in SPLITS:
    img_dir = os.path.join(DATA_ROOT, 'img', split)
    mask_dir = os.path.join(DATA_ROOT, 'labels', split)
    
    if not os.path.exists(img_dir):
        continue
        
    img_files = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
    
    for f_name in tqdm(img_files, desc=f"Аудит {split}"):
        base_name, _ = os.path.splitext(f_name)
        mask_path = os.path.join(mask_dir, base_name + '.png')
        img_path = os.path.join(img_dir, f_name)
        
        if not os.path.exists(mask_path):
            continue
            
        # 1. Проверка физических размеров
        with Image.open(img_path) as img:
            img_w, img_h = img.size
        with Image.open(mask_path) as mask:
            mask_w, mask_h = mask.size
            
        if (img_w != mask_w) or (img_h != mask_h):
            report[split]['mismatch_size'].append((f_name, f"Img: {img_w}x{img_h}, Mask: {mask_w}x{mask_h}"))
            continue
            
        # 2. Анализ связных областей (поиск шума и контуров)
        mask_arr = np.array(Image.open(mask_path))
        
        # Проверяем каждый целевой класс (1 и 2)
        has_noise = False
        for class_id in TARGET_CLASSES:
            binary_mask = (mask_arr == class_id).astype(np.uint8)
            if np.sum(binary_mask) == 0:
                continue
                
            # Ищем изолированные объекты этого класса
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
            
            # Проверяем площади найденных островков (минуя фон под индексом 0)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < MIN_OBJECT_AREA:
                    has_noise = True
                    break
                    
        if has_noise:
            report[split]['noisy_masks'].append(f_name)

# --- ИТОГОВЫЙ ОТЧЕТ ---
print("\n=== ОТЧЕТ О СКРЫТЫХ ОШИБКАХ ===")
for split in SPLITS:
    print(f"\nСплит [{split.upper()}]:")
    print(f"  - Несовпадений размеров (Картинка VS Маска): {len(report[split]['mismatch_size'])}")
    if report[split]['mismatch_size']:
        print(f"    Примеры: {report[split]['mismatch_size'][:3]}")
        
    print(f"  - Масок с «шумной» или битой разметкой (одиночные пиксели): {len(report[split]['noisy_masks'])}")
    if report[split]['noisy_masks']:
        print(f"    Примеры файлов: {report[split]['noisy_masks'][:5]}")
