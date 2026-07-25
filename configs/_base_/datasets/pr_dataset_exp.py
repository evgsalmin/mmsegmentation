# dataset settings
# Указываем тип датасета, это имя класса, который мы написали. 
# Так как мы зарегистрировали его с помощью декоратора @DATASETS.register_module()
# Теперь mmsegmentation может создавать его экземпляры, читая тип из конфига 
dataset_type = 'PrDataset'
data_root = "./dataset"
crop_size = (256, 256)

# ==== Определяем обучающий пайплайн данных ======
# Напомним, что датасет исходно отвечает только за то, чтобы распознать структуру данных
# Все остальные операции мы передаём как пайплайн
# Здесь у нас минимальный набор для обучения
# Чтение картинки и разметки и организация их в формате, который подходит для обучения
# Это базовый пайплайн, при реальном использовании вы можете добавить какие-то этапы 
# между LoadAnnotations и PackSegInputs
# ==== Определяем обуающий пайплайн данных ======
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # Аугментации 
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    
    # ДОБАВЛЕНО: Случайный поворот от -10 до +10 градусов
    dict(
        type='RandomRotate',
        degree=(-10, 10),      # Диапазон углов поворота
        prob=0.5,              # Вероятность применения к картинке (50%)
        pad_val=0,             # Чем заливать пустые углы на картинке (черный цвет)
        seg_pad_val=0          # Чем заливать пустые углы на маске (класс 0 / bg)
    ),
    
    dict(type='PhotoMetricDistortion'),
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
    batch_size=16,
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
    dataset=test_dataset
)


# Здесь же в пайплайне данных создаются объекты для подсчета метрик
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator 