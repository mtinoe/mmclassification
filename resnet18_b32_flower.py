# 模型配置
_base_ = ['../_base_/models/resnet18.py']
model = dict(
    head=dict(
        num_classes=5,
        topk = (1,)
    ))
#数据配置
_base_ =['../_base_/models/resnet18.py','../_base_/datasets/imagenet_bs32.py','../_base_/default_runtime.py']

data = dict(
# 根据实验环境调整每个 batch_size 和 workers 数量
    samples_per_gpu = 32,
    workers_per_gpu = 2,
# 指定训练集路径
    train = dict(
        data_prefix ='data/flower/train',
        ann_file ='data/flower/train.txt',
        #ann_file = None,
        classes ='data/flower/classes.txt'
    ),
# 指定验证集路径
    val = dict(
        data_prefix ='data/flower/val',
        #ann_file ='data/flower/val.txt',
        ann_file=None,
        classes ='data/flower/classes.txt'
    )
)
# 定义评估⽅法
evaluation = dict(metric_options={'topk': (1, )})

#修改训练策略设置
# 优化器
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# 学习率策略
lr_config = dict(
    policy='step',
    step=[1])

runner = dict(type='EpochBasedRunner', max_epochs=2)

load_from = 'checkpoints/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'