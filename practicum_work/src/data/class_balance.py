import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

DATA_ROOT = 'dataset/'
SPLITS = ['train', 'val', 'test']
NUM_CLASSES = 3  # Классы: 0, 1, 2

print("🔎 Расчет попиксельного баланса классов по всему датасету...")

for split in SPLITS:
    mask_dir = os.path.join(DATA_ROOT, 'labels', split)
    if not os.path.exists(mask_dir):
        continue
        
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    total_pixels = 0
    
    for f_name in tqdm(mask_files, desc=f"Подсчет пикселей в {split}"):
        mask_path = os.path.join(mask_dir, f_name)
        try:
            mask_arr = np.array(Image.open(mask_path))
            
            # Считаем уникальные значения на этой маске
            unique, counts = np.unique(mask_arr, return_counts=True)
            for cl, cnt in zip(unique, counts):
                if cl in class_counts:
                    class_counts[cl] += cnt
                    total_pixels += cnt
        except Exception as e:
            print(f"Ошибка при чтении {f_name}: {e}")
            
    print(f"\n📊 --- Распределение классов в [{split.upper()}] ---")
    if total_pixels == 0:
        print("Нет данных или маски пустые.")
        continue
        
    for class_id, count in class_counts.items():
        percentage = (count / total_pixels) * 100
        class_name = "Фон (0)" if class_id == 0 else f"Класс {class_id}"
        print(f"  - {class_name}: {count:,} px ({percentage:.3f}%)")
    print("-" * 50)
