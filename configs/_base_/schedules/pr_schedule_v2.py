epoch_num = 300

# Подготовим оптимайзер, используем дефолтные параметры 
optimizer = dict(type='AdamW', lr=0.0003, weight_decay=0.01)
# OptimWrapper это обертка над оптимизатором, нужна чтобы стандартные оптимизаторы 
# и более сложные реализации из mmsegmentation имели один интерфейс совместимый с Runner 
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# Определяем распорядок LR
param_scheduler = [
    # Линейный разогрев на первых 5 эпохах
    dict(
        type='LinearParamScheduler',
        param_name='lr',    # ДОБАВЛЕНО: явно указываем, что меняем Learning Rate
        start_factor=1e-6,
        by_epoch=True,     
        begin=0,
        end=5
    ),
    # Плавное косинусное затухание с 5 по 300 эпоху
    dict(
        type='CosineAnnealingLR',
        
        eta_min=1e-6,
        by_epoch=True,     
        begin=5,
        end=epoch_num
    )
]

# Определяем спецфику обучающего и тренироворочного циклов
# У mmsegmentation много вариаций, как организовать обучающий цикл.
# Мы используем наиболее привычную, когда одна эпоха — это один проход по датасету
# Есть также обучающие циклы на основе итераций, в них одна эпоха — это какое-то число итераций 
# они используется с семплером, который бесконечно зацикливает датасет 

# Если вы будете смотреть на стандратные конфиги, там обычно именно такой подход
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch_num)
# С валидационным и тестовым попроще: берём стандартные реализации
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# Указываем хуки для дополнительных возможностей, про которые мы говорили ранее 
# Обратите внимание, что аргумент определяется как default_hook 
# Дело в том, что mmsegmentation делит хуки ещё на две группы: default и custom
# default — это наиболее базовые хуки, и на первых порах мы ограничимся именно ими 
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=20),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=10, draw=True)
)