_base_ = [
    '../../_base_/models/faster-rcnn_r50_fpn.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

find_unused_parameters = True

BATCH_SIZE = 4
NUM_WORKERS = 1

DATASET_TYPE = 'VOCDataset'
DATA_ROOT = 'dataset_root_path/daytime_clear'
CLASSES = ("bus", "bike", "car", "motor", "person", "rider", "truck")
META_INFO = {'classes': CLASSES}

TRAIN_ANNO = ''
VAL_ANNO = ''
TEST_ANNO = ''

EPOCH = 12
LR = 0.02


model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')), 
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(
            type='NASBBoxHead',
            num_classes=7,
            is_search=True,
            arch_path=None,
            cell_cfg=dict(
                C=36,
                steps=4,
                multiplier=4,
                stem_multiplier=3),
            loss_cls=dict(
                type='CrossEntropyGLoss', use_sigmoid=False, loss_weight=1.0, penalty_weight=0.5),
            loss_bbox=dict(type='L1GLoss', loss_weight=1.0, penalty_weight=0.5))))

# dataset settings
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        metainfo=META_INFO,
        ann_file='VOC2007/ImageSets/Main/train.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=False,
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        metainfo=META_INFO,
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

# Pascal VOC2007 uses `11points` as default evaluate mode, while PASCAL
# VOC2012 defaults to use 'area'.
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=EPOCH, val_interval=1)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=LR, momentum=0.9, weight_decay=0.0001))
