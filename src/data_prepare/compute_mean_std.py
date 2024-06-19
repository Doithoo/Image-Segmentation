import os
from PIL import Image
import numpy as np


def main():
    img_channels = 3
    img_dir = "../../data/VOCdevkit/VOC2012/JPEGImages/"
    mask_dir = "../../data/VOCdevkit/VOC2012/SegmentationClass/"
    image_name_txt = "../../data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt"  # 2913张用于分割

    assert os.path.exists(image_name_txt), f"image path txt: '{image_name_txt}' does not exist."
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."
    assert os.path.exists(mask_dir), f"roi dir: '{mask_dir}' does not exist."

    with open(image_name_txt, "r") as f:
        img_name_list = f.read().splitlines()

    img_name_list = [i + ".jpg" for i in img_name_list]

    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".png"))
        img = np.array(Image.open(img_path)) / 255.  # 归一化处理
        mask = np.array(Image.open(mask_path).convert('L'))

        img = img[mask > 0]  # 找出图像中目标区域
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = np.round(cumulative_mean / len(img_name_list), 3)
    std = np.round(cumulative_std / len(img_name_list), 3)

    mean = tuple(mean.tolist())
    std = tuple(std.tolist())

    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    main()

# voc2012 dataset中用于语义分割的2193张图片的均值和标准差
# mean: [0.41163767 0.37499266 0.34729656]
# std: [0.23392912 0.22477559 0.22129112]
# mean: (0.412, 0.375, 0.347)
# std: (0.234, 0.225, 0.221)
