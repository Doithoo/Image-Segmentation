# @Author  : James
# @File    : validation.py
# @Description :
import os
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models

from data_prepare.my_dataset import VOCSegmentation
from models.fcn import fcn_resnet50

from utils import ConfusionMatrix

def test_transforms(input_size, norm_mean, norm_std):

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


def main(args):

    # 设置随机数种子，确保结果可重复
    torch.manual_seed(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 数据集准备
    norm_mean = (0.412, 0.375, 0.347)
    norm_std = (0.234, 0.225, 0.221)

    class_names = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    test_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=test_transforms(input_size=args.input_size,
                                                             norm_mean=norm_mean,
                                                             norm_std=norm_std),
                                  txt_name="val.txt")

    test_num = len(test_dataset)
    print(f"using {test_num} images for testing.")

    test_loader = DataLoader(test_dataset,
                            batch_size=4,
                            pin_memory=True,
                            collate_fn=test_dataset.collate_fn)

    # 构建模型
    model = fcn_resnet50(aux=args.aux, num_classes=num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'])
    model.to(device)

    confusion = ConfusionMatrix(num_classes=num_classes, class_names=class_names)

    model.eval()
    with torch.no_grad():
        for image, target in test_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confusion.update(output.argmax(1).flatten().cpu().numpy(), target.flatten().cpu().numpy())

        confusion.reduce_from_all_processes()
        confusion.summary()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn testing")

    parser.add_argument("--data-path", default="../data/", help="VOCdevkit root")
    parser.add_argument("--input-size", default=224, type=int, help="input_size")
    parser.add_argument("--weights", default="", help="load weight")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda:0", help="training device")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weight"):
        os.mkdir("./save_weight")

    main(args)
