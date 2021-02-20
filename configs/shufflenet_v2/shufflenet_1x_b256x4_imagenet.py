_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/models/shufflenet_v2_1x.py",
    "../_base_/datasets/imagenet_bs64_pil_resize.py",
]

data = dict(samples_per_gpu=256, workers_per_gpu=4)  # 4gpus
optimizer = dict(
    type="SGD", lr=0.5, momentum=0.9, weight_decay=0.00004, paramwise_cfg=dict(norm_decay_mult=0)
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy="poly",
    min_lr=0.0001,
    # False --> iter-based poly policy
    by_epoch=False,
    warmup="constant",
    warmup_iters=5000,
)
evaluation = dict(interval=60, metric="accuracy")
runner = dict(type="EpochBasedRunner", max_epochs=240)
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
