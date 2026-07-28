import os
import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_ROOT = 'dataset/'
SPLITS = ['train', 'val', 'test']

global_unique_classes = set()
split_classes = {}

print("🔎 Сканируем маски для определения реального количества классов...")

for split in SPLITS:
    mask_dir = os.path.join(DATA_ROOT, 'labels', split)
    if not os.path.exists(mask_dir):
        continue
        
    files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]
    split_set = set()
    
    for f_name in tqdm(files, desc=f"Анализ {split}"):
        mask_path = os.path.join(mask_dir, f_name)
        
        # Читаем маску как Grayscale numpy-массив
        mask_arr = np.array(Image.open(mask_path))
        unique_in_file = np.unique(mask_arr)
        
        split_set.update(unique_in_file)
        global_unique_classes.update(unique_in_file)
        
    split_classes[split] = sorted(list(split_set))

print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА КЛАССОВ ===")
print(f"Все уникальные ID пикселей, найденные в датасете: {sorted(list(global_unique_classes))}")
print(f"Итого потенциальных классов (включая фон): {len(global_unique_classes)}")

for split, classes in split_classes.items():
    print(f"  - В сплите [{split.upper()}] присутствуют ID: {classes}")
