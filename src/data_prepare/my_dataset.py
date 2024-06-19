# @Author  : James
# @File    : my_dataset.py
# @Description :
import os
from torch.utils.data import Dataset
from PIL import Image

# 自定义数据集
class VOCSegmentation(Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()

        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"

        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), f"path {root} does not exist."

        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), f"file '{txt_path}' does not exist."

        with open(txt_path, "r") as f:
            img_name_list = f.read().splitlines()

        self.images = [os.path.join(image_dir, f"{x}.jpg") for x in img_name_list]
        self.masks = [os.path.join(mask_dir, f"{x}.png") for x in img_name_list]

        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))

        # 对于数据集中图像大小不一致的情况，进行填充
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    train_dataset = VOCSegmentation(voc_root="../../data/")
    print(len(train_dataset))
    image, target = train_dataset[0]
    image.show()
    target.show()
    print(image.size)  # (weight, height)(500, 281)
    print(target.size)


