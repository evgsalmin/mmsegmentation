import os
import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_ROOT = 'dataset/'
SPLITS = ['train', 'val', 'test']

empty_masks_report = {}

print("🔎 Сканирование масок на наличие пустых сэмплов...")

for split in SPLITS:
    mask_dir = os.path.join(DATA_ROOT, 'labels', split)
    if not os.path.exists(mask_dir):
        continue
        
    files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]
    empty_files = []
    
    for f_name in tqdm(files, desc=f"Анализ {split}"):
        mask_path = os.path.join(mask_dir, f_name)
        
        # Загружаем маску
        mask_arr = np.array(Image.open(mask_path))
        
        # Проверяем, равен ли максимум нулю (значит, других классов нет)
        if mask_arr.max() == 0:
            empty_files.append(f_name)
            
    empty_masks_report[split] = empty_files

print("\n=== ОТЧЕТ ПО ПУСТЫМ МАСКАМ ===")
for split, empty_list in empty_masks_report.items():
    total_masks = len([f for f in os.listdir(os.path.join(DATA_ROOT, 'labels', split)) if f.lower().endswith('.png')])
    pct = (len(empty_list) / total_masks * 100) if total_masks > 0 else 0
    print(f"Сплит [{split.upper()}]: Найдено {len(empty_list)} из {total_masks} пустых масок ({pct:.2f}%)")
    if empty_list:
        print(f"  - Первые 5 пустых файлов: {empty_list[:5]}")
