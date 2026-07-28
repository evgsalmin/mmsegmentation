import os

DATA_ROOT = 'dataset/'
SPLITS = ['train', 'val', 'test']

for split in SPLITS:
    img_dir = os.path.join(DATA_ROOT, 'img', split)
    mask_dir = os.path.join(DATA_ROOT, 'labels', split)
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"⚠️ Папка для сплита {split} не найдена. Пропускаем.")
        continue
        
    # Читаем файлы и убираем расширения для точного сравнения имен
    img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if not f.startswith('.')}
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir) if not f.startswith('.')}
    
    # Ищем нестыковки
    images_without_masks = set(img_files.keys()) - set(mask_files.keys())
    masks_without_images = set(mask_files.keys()) - set(img_files.keys())
    
    print(f"\n=== Проверка сплита: {split.upper()} ===")
    print(f"Всего картинок: {len(img_files)} | Всего масок: {len(mask_files)}")
    
    if images_without_masks:
        print(f"❌ Картинки без масок ({len(images_without_masks)} шт.):")
        for name in list(images_without_masks)[:5]: # выведем первые 5
            print(f"  - {img_files[name]}")
            
    if masks_without_images:
        print(f"❌ Маски без картинок ({len(masks_without_images)} шт.):")
        for name in list(masks_without_images)[:5]:
            print(f"  - {mask_files[name]}")
            
    if not images_without_masks and not masks_without_images:
        print("✅ Сплит полностью синхронизирован. Ошибок парности нет.")
