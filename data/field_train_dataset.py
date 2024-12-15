import sys
sys.path.append('/home/maria/Documents/projects/anomaly_detection_study')

import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import random
import imgaug.augmenters as iaa # traditional argumentations

from dataau.perlin_noise import perlin
from dataau.cutpaste import cutpaste
from dataau.realsyn.realsyn import RealSyn


class FieldTrainDataset(Dataset):

    def __init__(self, image_dir, resize_shape=(256, 256*2), method='perlin', anomaly_source_dir=None):

        self.image_dir = image_dir
        self.anomaly_source_dir = anomaly_source_dir
        self.resize_shape = resize_shape
        self.method = method

        if (self.anomaly_source_dir is None) or (self.method == 'cutpaste'):
            self.anomaly_source_dir = image_dir

        self.image_paths = sorted(glob.glob(f"{self.image_dir}/*/*.png") + glob.glob(f"{self.image_dir}/*/*.jpg") + glob.glob(f"{self.image_dir}/*/*.jpeg"))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        self.anomaly_source_paths = sorted(glob.glob(f"{self.anomaly_source_dir}/*/*.png") + glob.glob(f"{self.anomaly_source_dir}/*/*.jpg") + glob.glob(f"{self.anomaly_source_dir}/*/*.jpeg"))

        if len(self.anomaly_source_paths) == 0:
            raise ValueError(f"No anomaly source images found in {anomaly_source_dir}")

        if self.method == 'realsyn':
            self.anomaly_source_paths = [path for path in self.anomaly_source_paths if path.endswith('_source.png')]

        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
           # iaa.Affine(rotate=(-45, 45))
        ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def randAugmenter(self):
        aug_indices = np.random.choice(len(self.augmenters), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[i] for i in aug_indices])
        return aug

    def augment_image(self, image, anomaly_source_path, method):

        aug = self.randAugmenter()

        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB)

        if method == 'perlin':
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            anomaly_source_img = cv2.resize(anomaly_source_img, (self.resize_shape[1], self.resize_shape[0]))
            anomaly_img_augmented = aug(image=anomaly_source_img)

            augmented_image, perlin_thr = perlin(anomaly_img_augmented, image, self.resize_shape)

            no_anomaly = torch.rand(1).item()
            if no_anomaly > 0.5:
                return image.astype(np.float32), np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
            else:
                augmented_image = augmented_image.astype(np.float32)

                mask = perlin_thr.astype(np.float32)
                augmented_image = mask * augmented_image + (1 - mask) * image
                has_anomaly = 1.0 if np.sum(mask) > 0 else 0.0
                return augmented_image, mask, np.array([has_anomaly], dtype=np.float32)

        elif method == 'cutpaste':
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            anomaly_source_img = cv2.resize(anomaly_source_img, (self.resize_shape[1], self.resize_shape[0]))
            anomaly_img_augmented = aug(image=anomaly_source_img)
            augmented_image, mask = cutpaste(anomaly_img_augmented, image)

            no_anomaly = torch.rand(1).item()
            if no_anomaly > 0.5:
                return image.astype(np.float32), np.zeros_like(mask, dtype=np.float32), np.array([0.0], dtype=np.float32)
            else:
                augmented_image = augmented_image.astype(np.float32)
                mask = mask.astype(np.float32)
                augmented_image = mask * augmented_image + (1 - mask) * image
                has_anomaly = 1.0 if np.sum(mask) > 0 else 0.0
                return augmented_image, mask, np.array([has_anomaly], dtype=np.float32)

        elif method == 'realsyn':

            other_path = anomaly_source_path.replace('_source', '_mask')
            img = cv2.imread(other_path, cv2.IMREAD_GRAYSCALE)
            source_mask = (img > 127).astype(np.uint8)

            #print(f"source_mask={source_mask.shape}")

            realsyn_method = random.choice(['alpha', 'poisson', 'direct'])

            #augmented_image, mask = realsyn(anomaly_source_img, image, source_mask, self.resize_shape, realsyn_method)
            composer = RealSyn(source=anomaly_source_img, target=image, mask=source_mask, method=realsyn_method, resize_shape=self.resize_shape)
            augmented_image, mask = composer.run()
            
            #aau = torch.rand(1).item()
            #if aau > 0.5:
            #    augmented_image = aug(image=augmented_image)

            no_anomaly = torch.rand(1).item()
            if no_anomaly > 0.5:
                image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
                image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
                return image.astype(np.float32), np.zeros_like(mask, dtype=np.float32), np.array([0.0], dtype=np.float32)
            else:
                augmented_image = augmented_image.astype(np.float32)/255.
                mask = mask.astype(np.float32)
                has_anomaly = 1.0 if np.sum(mask) > 0 else 0.0
                return augmented_image, mask, np.array([has_anomaly], dtype=np.float32)

        elif method == None:
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            anomaly_source_img = cv2.resize(anomaly_source_img, (self.resize_shape[1], self.resize_shape[0]))
            augmented_image = aug(image=anomaly_source_img)

            augmented_image = augmented_image.astype(np.float32)
            mask = np.zeros(self.resize_shape, dtype=np.float32)
            mask = np.expand_dims(mask, axis=2)
            augmented_image = mask * augmented_image + (1 - mask) * image
            has_anomaly = 1.0 if np.sum(mask) > 0 else 0.0
            return augmented_image, mask, np.array([has_anomaly], dtype=np.float32)


    def transform_image(self, image_path, anomaly_source_path, method):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if method != 'realsyn':
            image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))

        if (torch.rand(1).item() > 0.7) and (method != 'realsyn'):
            image = self.rot(image=image)

        #image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path, method)
        image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))

        if method == 'realsyn':
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0

        #print("before")
        #print(augmented_image.shape, anomaly_mask.shape, has_anomaly.shape)

        image = np.transpose(image, (2, 0, 1))
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))

        #print("after")
        #print(augmented_image.shape, anomaly_mask.shape, has_anomaly.shape)

        return image, augmented_image, anomaly_mask, has_anomaly


    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()

        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx],
                                                                           self.anomaly_source_paths[anomaly_source_idx], self.method)

        sample = {
            'image': image,
            'augmented_image': augmented_image,
            'anomaly_mask': anomaly_mask,
            'has_anomaly': has_anomaly,
            'idx': idx
        }

        return sample

