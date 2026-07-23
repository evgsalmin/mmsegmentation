# dataset settings
# Указываем тип датасета, это имя класса, который мы написали. 
# Так как мы зарегистрировали его с помощью декоратора @DATASETS.register_module()
# Теперь mmsegmentation может создавать его экземпляры, читая тип из конфига 
dataset_type = 'PrDataset'
data_root = "./dataset"

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
    # Аугментацции 
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomRotFlip', degree=(-45, 45)),
    dict(type='RandomCutOut', prob=0.4, n_holes=(7, 15), cutout_ratio=(0.1, 0.15)),
    dict(type='Albu', transforms=[dict(type="GridDistortion", num_steps=10, p=1)]),
    # ===
    dict(type='PackSegInputs')
]
train_dataset=dict(
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
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs")
]
test_pipeline = val_pipeline

val_dataset = dataset=dict(
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
test_dataset = dataset=dict(
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