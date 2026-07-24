# dataset settings
# Указываем тип датасета, это имя класса, который мы написали. 
# Так как мы зарегистрировали его с помощью декоратора @DATASETS.register_module()
# Теперь mmsegmentation может создавать его экземпляры, читая тип из конфига 
dataset_type = 'PrDataset'
data_root = "./dataset"
crop_size = (256, 256)

# ==== Обучающий пайплайн с агрессивной регуляризацией (Гипотеза 2) ======
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    
    # 1. Геометрические изменения (масштабирование и кроп)
    # Диапазон scale_factor от 0.5 до 1.5 позволяет имитировать разное удаление объектов
    dict(type='RandomResize', scale=(256, 256), ratio_range=(0.5, 1.5), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    
    # 2. Искажения цвета (яркость, контраст, насыщенность)
    dict(type='PhotoMetricDistortion'),
    
    # 3. Продвинутые аугментации через Albumentations (Разрушение текстур)
    dict(
        type='Albu',
        transforms=[
            # Случайный выбор между размытием и добавлением гауссова шума
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='GaussNoise', var_limit=(10.0, 50.0), p=1.0),
                ],
                p=0.5
            ),
            # CutOut / CoarseDropout: затирает случайные квадраты на картинке.
            # Позволяет модели находить кошек/собак, даже если они частично перекрыты.
            dict(
                type='CoarseDropout',
                max_holes=4,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=0,          # Затираем нулями (черный цвет на картинке)
                mask_fill_value=0,   # Игнорируем затертую область в лоссе (seg_pad_val)
                p=0.5
            )
        ]
    ),
    
    # 4. Финальная упаковка тензоров в формат модели
    dict(type='PackSegInputs')
]

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/train',
        seg_map_path='labels/train'),
    pipeline=train_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)

train_dataloader = dict(
    batch_size=16,             # Размер батча 16 отлично подходит для AdamW
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset   
)


# ==== Определяем валидационный пайплайн данных ======
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs")
]
test_pipeline = val_pipeline

val_dataset =dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/val',
        seg_map_path='labels/val'),
    pipeline=val_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset
)


# ==== Определяем тестовый пайплайн данных ======
test_dataset =dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='img/test',
        seg_map_path='labels/test'),
    pipeline=test_pipeline,
    img_suffix=".jpg",
    seg_map_suffix=".png"
)
test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset
)


# Здесь же в пайплайне данных создаются объекты для подсчета метрик
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator 