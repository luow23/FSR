import os
# import tarfile
from PIL import Image
# import urllib.request
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imgaug.augmenters as iaa
import glob
from datasets.perlin import rand_perlin_2d_np
import numpy as np
import cv2
import random
# CLASS_NAMES = ['cable', 'carpet', 'grid', 'pill',
#                      'tile', 'toothbrush', 'wood', 'zipper']
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                     'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                     'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
class UniTrainData(Dataset):
    def __init__(self, opt, resize=256, anomaly_sourec_path=r'D:\IMSN-LW\dataset\image_net'):
        self.images = []
        root = opt.data_root
        self.anomaly_source_paths = sorted(glob.glob(anomaly_sourec_path + "/*.JPEG"))
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        self.transform_x = transforms.Compose([
            transforms.Resize(resize, Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std= [0.229, 0.224, 0.225])])
        for v in CLASS_NAMES:
            path_list = os.path.join(root, v, 'train')
            imgs = GetFiles(path_list, ["JPG", "jpg", "bmp", "png"])
            imgs = [img for img in imgs]
            self.images.extend(imgs)
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):

        random_nature_img_name = random.sample(anomaly_source_path, 1)[0]
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(random_nature_img_name)
        # cv2.imwrite('luowei4.jpg', anomaly_source_img)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize, self.resize))
        cv2.imwrite('luowei3.jpg', anomaly_source_img)

        anomaly_img_augmented = aug(image=anomaly_source_img)
        cv2.imwrite('luowei4.jpg', anomaly_img_augmented)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize, self.resize), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0]

        augmented_image = image * (1 - perlin_thr) +  img_thr*perlin_thr

        # no_anomaly = torch.rand(1).numpy()[0]
        # if no_anomaly > 0.8:
        #     image = image.astype(np.float32)
        #     return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)

            # augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        # augmented_image = msk * augmented_image + (1 - msk) * image
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly = 0.0
        return augmented_image, msk, np.array([has_anomaly], dtype=np.float32),

    def random_anomaly(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize, self.resize))
        # image = image/255.0

        image = np.array(image).astype(np.float32) / 255.0
        # print(image.shape)
        aug_img, aug_mask, aug_label = self.augment_image(image, self.anomaly_source_paths)
        # print(aug_img.shape)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        aug_img = cv2.cvtColor(np.uint8(aug_img*255), cv2.COLOR_BGR2RGB)
        # aug_img = np.transpose(aug_img, (2, 0, 1))
        # aug_mask = np.transpose(aug_mask, (2, 0, 1))
        return aug_img, aug_mask, aug_label
    def __getitem__(self, item):
        x = self.images[item]
        # x = Image.open(img_path).convert('RGB').resize((256, 256))
        # img = self.transform_x(x)
        # x, y, mask, name = self.x[idx], self.y[idx], self.mask[idx], self.name[idx]
        aug_x, aug_mask, aug_label = self.random_anomaly(x)
        aug_x = Image.fromarray(np.uint8(aug_x))
        aug_x = self.transform_x(aug_x)
        aug_mask = aug_mask.reshape(aug_mask.shape[0], aug_mask.shape[1])
        aug_mask = Image.fromarray(np.uint8(aug_mask * 255))
        aug_mask = self.transform_mask(aug_mask)
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)
        # if self.train_stage == 1:
        #     return x, y, mask
        # elif self.train_stage == 2:
        return x, aug_x, aug_mask, aug_label
        return img
    def __len__(self):
        return len(self.images)

def GetFiles(file_dir, file_type, IsCurrent=False):
    file_list = []
    for parent, dirnames, filenames in os.walk(file_dir):
        for filename in filenames:
            for type in file_type:
                if filename.endswith(('.%s' % type)):
                    file_list.append(os.path.join(parent, filename))

        if IsCurrent == True:
            break
    return file_list