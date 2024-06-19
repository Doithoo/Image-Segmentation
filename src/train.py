# @Author  : James
# @File    : train.py
# @Description :
import os
import time
import datetime
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models

from data_prepare.my_dataset import VOCSegmentation
from models.fcn import fcn_resnet50
from utils import train_one_epoch, evaluate, custom_lr_scheduler
def train_transforms(input_size,  norm_mean, norm_std, hflip_prob=0.5):
    min_size = int(1.2 * input_size)
    max_size = int(2.0 * input_size)
    rand_resize = random.randint(min_size, max_size)

    return transforms.Compose([
        transforms.Resize(rand_resize),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(p=hflip_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


def valid_transforms(input_size, norm_mean, norm_std):
    min_size = int(1.2 * input_size)
    max_size = int(2.0 * input_size)
    rand_resize = random.randint(min_size, max_size)

    return transforms.Compose([
        transforms.Resize(rand_resize),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


def create_model(aux_enable, num_classes, pretrain=False):
    model = fcn_resnet50(aux=aux_enable, num_classes=num_classes)

    if pretrain:
        weights_dict = torch.load("./pretrained_weight/fcn_resnet50_coco.pth", map_location='cpu') # 载入预训练权重到CPU中

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys) # 缺失的权重
            print("unexpected_keys: ", unexpected_keys) # 没有用到的权重

    return model


def main(args):
    # 设置随机数种子，确保结果可重复
    torch.manual_seed(1)

    torch.backends.cudnn.benchmark = True  # 加快训练

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    results_file = f'results{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'

    #数据集准备
    norm_mean = (0.412, 0.375, 0.347)
    norm_std = (0.234, 0.225, 0.221)
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=train_transforms(input_size=args.input_size,
                                                                norm_mean=norm_mean,
                                                                norm_std=norm_std),
                                    txt_name="train.txt")

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=valid_transforms(input_size=args.input_size,
                                                              norm_mean=norm_mean,
                                                              norm_std=norm_std),
                                  txt_name="val.txt")

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print(f"using {train_num} images for training, {val_num} images for validation.")

    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) # 设定合适的线程数

    train_loader = DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               pin_memory=True,
                               collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset,
                             batch_size=4,
                             pin_memory=True,
                             collate_fn=val_dataset.collate_fn)

    print(f"总共有{args.epochs * len(train_loader)}steps.")

    model = create_model(aux_enable=args.aux_enable, num_classes=num_classes, pretrain=args.pretrain)
    model.to(device)

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]}, # 取出没有冻结的权重
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if args.aux_enable: # 如果启用辅助分类器的话，加载相应的权重
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None # 混合精度训练

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)  # 设置学习率下降策略
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    lr_scheduler = custom_lr_scheduler(optimizer,
                                       num_step_each_epoch=len(train_loader),  # 每个epoch内的步骤数，用于将epoch转换为总步数。
                                       epochs=args.epochs,
                                       warmup=args.enable_warmup,
                                       warmup_epochs=args.warmup_epochs,
                                       )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    print('Start Training...')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_loss_each_epoch, lr = train_one_epoch(model,
                                        optimizer,
                                        train_loader,
                                        device,
                                        epoch,
                                        lr_scheduler=lr_scheduler,
                                        print_freq=args.print_freq,
                                        scaler=scaler)

        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)

        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {train_loss_each_epoch:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, f"save_weights/model_{epoch}.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"training time {total_time_str}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--data-path", default="../data/", help="VOCdevkit root")
    parser.add_argument("--input-size", default=224, type=int, help="input_size")
    parser.add_argument("--num-classes", default=20, type=int)

    parser.add_argument("--aux-enable", default=False, type=bool, help="auxilier loss")  # 是否使用辅助分类器
    parser.add_argument("--pretrain", default=False, type=bool, help="load pretrained weights")

    parser.add_argument("--device", default="cuda:0", help="training device") # 默认使用第一块gpu,没有gpu就是用cpu
    parser.add_argument("-b", "--batch-size", default=4, type=int) # 根据gpu显存进行设置
    parser.add_argument("--enable-warmup", default=True, type=bool, help="enable warmup")
    parser.add_argument("--warmup-epochs", default=2, type=int, help="warmup epochs")
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weight"):
        os.mkdir("./save_weight")

    main(args)
