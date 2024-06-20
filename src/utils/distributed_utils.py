# @Author  : James
# @File    : distributed_utils.py
# @Description :
from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist
import numpy as np

# 它是一个用于创建和管理表格的简单库，可以帮助我们以美观的方式在控制台中打印表格数据。
from prettytable import PrettyTable

import errno
import os


__all__ = [
    "SmoothedValue",
    "ConfusionMatrix",
    "MetricLogger",
    "mkdir",
    "setup_for_distributed",
    "is_dist_avail_and_initialized",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "save_on_master",
    "init_distributed_mode",
]

class SmoothedValue(object):
    """
    SmoothedValue类设计用于监控并平滑处理一系列数值数据，
    主要应用于机器学习训练过程中的指标追踪（如损失、准确率等），
    旨在提供窗口内平滑值、全局平均值以及其它统计信息。
    """

    def __init__(self, window_size=20, fmt=None):
        """
        :param window_size: 窗口大小，默认为20，表示用于计算平滑值的最近数据点数量。
        :param fmt: 输出格式字符串，默认为"{value:.4f} ({global_avg:.4f})"，
        用于格式化打印输出，展示当前值与全局平均值。
        """
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"

        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        更新跟踪的值，增加指定次数n的value到统计中。
        :param value:
        :param n:
        :return:
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        在分布式训练环境下同步count和total的值，但注意它不直接同步deque内的数据。
        只有当分布式环境可用并初始化时执行此操作。
        :return:
        """

        if not is_dist_avail_and_initialized():
            return

        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """
        返回deque中所有值的中位数，使用PyTorch计算后转换为Python标量。
        :return:
        """

        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """
        返回deque中所有值的平均值。
        :return:
        """

        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """
        返回整个序列的全局平均值，即total除以count。
        :return:
        """
        return self.total / self.count

    @property
    def max(self):
        """
        返回deque中的最大值。
        :return:
        """
        return max(self.deque)

    @property
    def value(self):
        """
        返回deque中的最后一个加入的值。
        :return:
        """
        return self.deque[-1]

    def __str__(self):
        """
        格式化输出当前跟踪的统计信息，包括中位数、平均值、全局平均值、最大值和当前值，依据初始化时设定的fmt格式。
        :return:
        """
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class ConfusionMatrix(object):
    def __init__(self, num_classes, class_names):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.class_names = class_names

    def update(self, preds, labels):
        k = (labels >= 0) & (labels < self.num_classes)
        for p, t in zip(preds[k], labels[k]):
            self.matrix[p, t] += 1


    def compute(self):

        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]

        ious = []

        for i in range(self.num_classes):
            TP = self.matrix[i, i]  # 对角线总数
            FP = np.sum(self.matrix[i, :]) - TP  # pred总数 - TP
            FN = np.sum(self.matrix[:, i]) - TP  # Truth - TP

            iou = round(TP / (TP + FP + FN), 3) if TP + FP + FN != 0 else 0.
            ious.append(iou)

        acc_global = sum_TP / np.sum(self.matrix)
        miou = np.nanmean(ious)

        return acc_global, miou, ious
    def summary(self):

        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]

        ious = []
        # precision, recall, f1-score, iou
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "F1-score", "IoU"]

        for i in range(self.num_classes):
            TP = self.matrix[i, i]  # 对角线总数
            FP = np.sum(self.matrix[i, :]) - TP  # pred总数 - TP
            FN = np.sum(self.matrix[:, i]) - TP  # Truth - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            F1_score = round(2 * TP / (np.sum(self.matrix) + TP - TN), 3) if np.sum(self.matrix) + TP - TN != 0 else 0.
            iou = round(TP / (TP + FP + FN), 3) if TP + FP + FN != 0 else 0.
            ious.append(iou)

            table.add_row([self.class_names[i], Precision, Recall, F1_score, iou])

        acc_global = sum_TP / np.sum(self.matrix)
        miou = np.nanmean(ious)

        print("the model's accuracy is ", acc_global)
        print("the model's mIoU is ", miou)
        print(table)

    def reduce_from_all_processes(self):
        """
        在多GPU分布式训练环境中，使用torch.distributed API来收集所有进程上的混淆矩阵，
        确保所有进程具有相同的汇总混淆矩阵。
        :return:
        """
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.matrix)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter # 分隔符

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = [f"{name}: {str(meter)}" for name, meter in self.meters.items()]
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = f':{len(str(len(iterable)))}d'

        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])

        MB = 1024.0 * 1024.0
        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            # 每多少个steps打印一次日志
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str}')


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif not hasattr(args, "rank"):
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
