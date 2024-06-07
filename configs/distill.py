_base_ = ['student.py']

# model settings
find_unused_parameters = False

# config settings
fea = True
logit = True

# method details
model = dict(
    _delete_ = True,
    type='DWPoseDistiller',
    teacher_pretrained = 'teacher.pth',
    teacher_cfg = 'teacher.py',
    student_cfg = 'student.py',
    distill_cfg = [dict(methods=[dict(type='FeaLoss',
                                       name='loss_fea',
                                       use_this = fea,
                                       student_channels = 768,
                                       teacher_channels = 1024,
                                       alpha_fea=0.00007,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='KDLoss',
                                       name='loss_logit',
                                       use_this = logit,
                                       weight = 0.1,
                                       )
                                ]
                        ),
                    ],
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
)

optim_wrapper = dict(
    clip_grad=dict(max_norm=1., norm_type=2))