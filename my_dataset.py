import numpy
import numpy as np
import torch.utils.data as data
import torch
import os
import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image


# 读取花分类数据集
class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


# 读取VOC数据集
class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        if txt_name == 'train.txt':
            print("the number of train picture: {}".format(len(self.images)))
        else:
            print("the number of validation picture: {}".format(len(self.images)))

        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
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
        # t = target.getpalette()
        # print(t)
        # target = np.array(target)
        # print(target.shape)
        # np.set_printoptions(threshold=np.inf)
        # print(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


# 读取Potsdam数据集
#[269014873, 263979133, 180194919, 18768491, 46854081, 337188503]
# CLASS = ['background','car','tree','low vegetation','building','impervious surfaces']
# colormatching={[255,0,0]:'background',[255, 255, 0]:'car',[0, 255, 0]:'tree',
# [0, 255, 255]:'low vegetation',[0, 0, 255]:'building',[255, 255, 255]:'impervious surfaces'}
# rgb和bgr有个转换
# colormatching={[0, 0, 255]:'building',[0, 255, 255]:'low vegetation',[0, 255, 0]:'tree',
# [255, 255, 0]:'car',[255,0,0]:'background',[255, 255, 255]:'impervious surfaces'}
class PotsdamSegmentation(data.Dataset):
    def __init__(self, voc_root, transforms=None, txt_name: str = "train", predict=False):
        super(PotsdamSegmentation, self).__init__()
        root = os.path.join(voc_root, txt_name)
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'images')
        mask_dir = os.path.join(root, 'labels')
        self.predict=predict

        dir_image = os.listdir(image_dir)
        dir_label = os.listdir(mask_dir)

        self.images=[]
        self.masks=[]
        for x in dir_image:
            self.images.append(os.path.join(image_dir, x))
            self.masks.append(os.path.join(mask_dir, x.replace('jpg','png')))
        self.images.sort()
        self.masks.sort()
        if txt_name == 'train':
            print("the number of train picture: {}".format(len(self.images)))
        else:
            print("the number of validation picture: {}".format(len(self.images)))

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
        # print(self.images[index])
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(target)
        # plt.show()
        # target1=np.asarray(target)
        # res = torch.empty((224, 224, 3))
        # for i in range(224):
        #     for j in range(224):
        #         if target1[i][j] == 4:
        #             res[i][j][0] = 255
        #             res[i][j][1] = 0
        #             res[i][j][2] = 0
        #         if target1[i][j] == 3:
        #             res[i][j][0] = 255
        #             res[i][j][1] = 255
        #             res[i][j][2] = 0
        #         if target1[i][j] == 2:
        #             res[i][j][0] = 0
        #             res[i][j][1] = 255
        #             res[i][j][2] = 0
        #         if target1[i][j] == 1:
        #             res[i][j][0] = 0
        #             res[i][j][1] = 255
        #             res[i][j][2] = 255
        #         if target1[i][j] == 0:
        #             res[i][j][0] = 0
        #             res[i][j][1] = 0
        #             res[i][j][2] = 255
        #         if target1[i][j] == 5:
        #             res[i][j][0] = 255
        #             res[i][j][1] = 255
        #             res[i][j][2] = 255
        # plt.imshow(res)
        # plt.show()
        # target=target[:,:,0]

        if self.transforms is not None:
            img, target = self.transforms(img, target)



        # print(target.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(img.permute(1,2,0))
        # plt.show()
        # plt.imshow(target)
        # plt.show()
        # torch.set_printoptions(profile="full")
        # print(target)


        # for i in range(target.shape[0]):
        #     for j in range(target.shape[1]):
        #         if target[i, j, 0] >0  and target[i, j, 1] == 0 and target[i, j, 2] == 0:
        #             target[i, j, 0] = 0
        #             continue
        #         if target[i, j, 0] >0 and target[i, j, 1] > 0 and target[i, j, 2] == 0:
        #             target[i, j, 0] = 1
        #             continue
        #         if target[i, j, 0] == 0 and target[i, j, 1] >0 and target[i, j, 2] == 0:
        #             target[i, j, 0] = 2
        #             continue
        #         if target[i, j, 0] == 0 and target[i, j, 1] >0 and target[i, j, 2] >0:
        #             target[i, j, 0] = 3
        #             continue
        #         if target[i, j, 0] == 0 and target[i, j, 1] == 0 and target[i, j, 2] >0:
        #             target[i, j, 0] = 4
        #             continue
        #         if target[i, j, 0] >0 and target[i, j, 1] >0 and target[i, j, 2] >0:
        #             target[i, j, 0] = 5
        #             continue
        #         print(target[i,j,:])
        # target = target[:, :, 0]
        # torch.set_printoptions(profile="full")
        # print(target.shape)
        if self.predict==True:
            filename=self.images[index]
            return img,filename,target
        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
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

    data_transform = {
        "train": T.Compose([T.ToTensor()],),
        }
    train_dataset = PotsdamSegmentation(voc_root='./dataset_vh_256',transforms=data_transform['train'])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=1,
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=train_dataset.collate_fn)

    channels_sum, channel_squared_sum = 0, 0
    num_batches = len(train_loader)

    #计算各个像素的个数
    # num_label=[0,0,0,0,0,0]
    # for data,label in train_loader:
    #     label=label.reshape(-1)
    #     for a in label:
    #         num_label[a]=num_label[a]+1
    # print(num_label)
    #[269014873, 263979133, 180194919, 18768491, 46854081, 337188503]

    #计算损失权重
    # num_label=[269014873, 263979133, 180194919, 18768491, 46854081, 337188503]
    # num=269014873+263979133+180194919+18768491+46854081+337188503
    # for i in num_label:
    #     print(num/float(i)/100)
    # 0.041484695160330404
    # 0.04227606884366879
    # 0.06193293385814059
    # 0.5946135999958654
    # 0.23818629587463255
    # 0.03309721387505315

    #计算均值和方差
    for data, _ in train_loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channel_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])

    mean = channels_sum / num_batches
    std = (channel_squared_sum / num_batches - mean ** 2) ** 0.5
    print(mean, std)
    #pot: tensor([0.3412, 0.3637, 0.3378]) tensor([0.1402, 0.1384, 0.1439])
    #vh: tensor([0.4696, 0.3191, 0.3144]) tensor([0.2148, 0.1551, 0.1481])