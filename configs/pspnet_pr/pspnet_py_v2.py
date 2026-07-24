_base_ = [
    '../_base_/models/pspnet_r50-d8.py', 
    '../_base_/datasets/pr_dataset_v2.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/pr_schedule_v2.py'
]

class_weight = [0.37, 3.28, 3.92]

visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend')     # сохраняем логи локально
    ]
)


# Определим размер входа 
input_size = (256, 256)
data_preprocessor = dict(
        size=input_size,
        # Стандартные веса ImageNet для нормализации
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255
    )

model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode="whole"),
    decode_head=dict(
        num_classes=3,
        dropout_ratio=0.5,
        pool_scales=(2, 4, 8, 12), 
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=class_weight
            ),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
        ]
    ),
    auxiliary_head=dict(
        num_classes=3,
        dropout_ratio=0.5,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=0.4,
                class_weight=class_weight 
            ),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
        ]
    ),
    
    init_cfg=dict(
        type='Pretrained',
        checkpoint='pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
    )
)