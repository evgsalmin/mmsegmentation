_base_ = [
    '../_base_/models/pspnet_r50-d8.py', 
    '../_base_/datasets/pr_dataset.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/pr_schedule.py'
]


visualizer = dict(
    type='Visualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),      # сохраняем логи локально
        
    ]
)


# Определим размер входа 
input_suze = (256, 256)
data_preprocessor = dict(size=input_suze)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode="whole"),
    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=1.0
            ),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
        ]
    ),
    auxiliary_head=dict(
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=1.0
            ),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)
        ]
    )
)