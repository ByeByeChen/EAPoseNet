auto_scale_lr = dict(base_batch_size=512)
backend_args = dict(backend='local')
base_lr = 0.004
codec = dict(
    input_size=(
        256,
        256,
    ),
    normalize=False,
    sigma=(
        5.66,
        5.66,
    ),
    simcc_split_ratio=2.0,
    type='SimCCLabel',
    use_dark=False)
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=2970,
        switch_pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(
                rotate_factor=60,
                scale_factor=[
                    0.75,
                    1.25,
                ],
                shift_factor=0.0,
                type='RandomBBoxTransform'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                transforms=[
                    dict(p=0.1, type='Blur'),
                    dict(p=0.1, type='MedianBlur'),
                    dict(
                        max_height=0.4,
                        max_holes=1,
                        max_width=0.4,
                        min_height=0.2,
                        min_holes=1,
                        min_width=0.2,
                        p=0.5,
                        type='CoarseDropout'),
                ],
                type='Albumentation'),
            dict(
                encoder=dict(
                    input_size=(
                        256,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        5.66,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
data_mode = 'topdown'
data_root = 'data/fly/'
dataset_type = 'FlyDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=10,
        max_keep_ckpts=2,
        rule='greater',
        save_best='PCK',
        type='CheckpointHook'),
    logger=dict(interval=1, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
max_epochs = 420
model = dict(
    backbone=dict(
        _scope_='mmpose',
        act_cfg=dict(type='SiLU'),
        arch='P5',
        channel_attention=True,
        channel_attention_type='SimAM',
        deepen_factor=0.67,
        expand_ratio=0.5,
        init_cfg=dict(),
        norm_cfg=dict(type='BN'),
        out_indices=(4, ),
        type='CSPNeXt_AKconv',
        use_akconv=False,
        use_dcnv4=False,
        use_sppelan=True,
        widen_factor=0.75),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            input_size=(
                256,
                256,
            ),
            normalize=False,
            sigma=(
                5.66,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        final_layer_kernel_size=7,
        gau_cfg=dict(
            act_fn='SiLU',
            drop_path=0.0,
            dropout_rate=0.0,
            expansion_factor=2,
            hidden_dims=256,
            pos_enc=False,
            s=128,
            use_rel_bias=False),
        in_channels=768,
        in_featuremap_size=(
            8,
            8,
        ),
        input_size=(
            256,
            256,
        ),
        loss=dict(
            beta=10.0,
            label_softmax=True,
            type='KLDiscretLoss',
            use_target_weight=True),
        out_channels=32,
        simcc_split_ratio=2.0,
        type='MagicHead'),
    test_cfg=dict(flip_test=True),
    type='TopdownPoseEstimator')
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=1500,
        begin=400,
        by_epoch=True,
        convert_to_iter_based=True,
        end=3000,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]
randomness = dict(seed=21)
resume = False
stage2_num_epochs = 30
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/fly_test.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/fly/',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='FlyDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(ann_file='data/fly/annotations/fly_test.json', type='CocoMetric'),
    dict(thr=0.2, type='PCKAccuracy'),
    dict(type='AUC'),
    dict(keypoint_indices=[
        1,
        2,
    ], norm_mode='keypoint_distance', type='NME'),
]
train_cfg = dict(by_epoch=True, max_epochs=3000, val_interval=10)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='annotations/fly_train.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/fly/',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(
                rotate_factor=80,
                scale_factor=[
                    0.6,
                    1.4,
                ],
                type='RandomBBoxTransform'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                transforms=[
                    dict(p=0.1, type='Blur'),
                    dict(p=0.1, type='MedianBlur'),
                    dict(
                        max_height=0.4,
                        max_holes=1,
                        max_width=0.4,
                        min_height=0.2,
                        min_holes=1,
                        min_width=0.2,
                        p=1.0,
                        type='CoarseDropout'),
                ],
                type='Albumentation'),
            dict(
                encoder=dict(
                    input_size=(
                        256,
                        256,
                    ),
                    normalize=False,
                    sigma=(
                        5.66,
                        5.66,
                    ),
                    simcc_split_ratio=2.0,
                    type='SimCCLabel',
                    use_dark=False),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='FlyDataset'),
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(
        rotate_factor=80,
        scale_factor=[
            0.6,
            1.4,
        ],
        type='RandomBBoxTransform'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=1.0,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                256,
                256,
            ),
            normalize=False,
            sigma=(
                5.66,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(
        rotate_factor=60,
        scale_factor=[
            0.75,
            1.25,
        ],
        shift_factor=0.0,
        type='RandomBBoxTransform'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        transforms=[
            dict(p=0.1, type='Blur'),
            dict(p=0.1, type='MedianBlur'),
            dict(
                max_height=0.4,
                max_holes=1,
                max_width=0.4,
                min_height=0.2,
                min_holes=1,
                min_width=0.2,
                p=0.5,
                type='CoarseDropout'),
        ],
        type='Albumentation'),
    dict(
        encoder=dict(
            input_size=(
                256,
                256,
            ),
            normalize=False,
            sigma=(
                5.66,
                5.66,
            ),
            simcc_split_ratio=2.0,
            type='SimCCLabel',
            use_dark=False),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/fly_test.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/fly/',
        pipeline=[
            dict(backend_args=dict(backend='local'), type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='FlyDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(ann_file='data/fly/annotations/fly_test.json', type='CocoMetric'),
    dict(thr=0.2, type='PCKAccuracy'),
    dict(type='AUC'),
    dict(keypoint_indices=[
        1,
        2,
    ], norm_mode='keypoint_distance', type='NME'),
]
val_pipeline = [
    dict(backend_args=dict(backend='local'), type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/fly_m_nodis_full'
