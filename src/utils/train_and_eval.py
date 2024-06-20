# @Author  : James
# @File    : train_and_eval.py
# @Description :
import torch
from torch import nn

from .distributed_utils import ConfusionMatrix, MetricLogger, SmoothedValue

__all__ = [
    "evaluate",
    "train_one_epoch",
    "custom_lr_scheduler",
]

def criterion(inputs, target):
    # 针对多个输出层，包括辅助输出层
    losses = {name: nn.functional.cross_entropy(x, target.squeeze(1).long(), ignore_index=255) for name, x in inputs.items()}

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes, class_names):
    model.eval()
    confmat = ConfusionMatrix(num_classes=num_classes, class_names=class_names)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    val_loss = 0.0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq=100, header=header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            val_loss += criterion(output, target).item()
            output = output['out']

            confmat.update(output.argmax(1).flatten().cpu().numpy(), target.long().flatten().cpu().numpy())

        confmat.reduce_from_all_processes()

    return confmat, val_loss / len(data_loader)


def safe_get_lr(optimizer):
    """安全地获取学习率，避免访问不存在的索引或键"""
    try:
        return optimizer.param_groups[0]["lr"]
    except IndexError:
        print("Warning: optimizer.param_groups is empty.")
        return 0.0
    except KeyError:
        print("Warning: 'lr' key not found in optimizer.param_groups[0].")
        return 0.0

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ") # 创建一个日志记录器, 使用空格分割
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]' # 从1开始
    lr = None
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        # 使用混合精度训练
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        # 安全地更新学习率日志
        lr = safe_get_lr(optimizer)
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def custom_lr_scheduler(optimizer,
                        num_step_each_epoch: int,  # 每个epoch内的步骤数，用于将epoch转换为总步数。
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,  # 表示warmup阶段持续的epoch数。
                        warmup_factor=1e-3): # warmup阶段初的学习率提升因子
    assert num_step_each_epoch > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        x: 当前迭代的步数steps = epoch * num_step
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is False or x > warmup_epochs * num_step_each_epoch:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step_each_epoch) / ((epochs - warmup_epochs) * num_step_each_epoch)) ** 0.9
        alpha = float(x) / (warmup_epochs * num_step_each_epoch)
        # warmup过程中lr倍率因子从warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)  # 允许用户自定义学习率更新规则
