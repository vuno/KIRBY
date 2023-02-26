import torchvision.datasets as dset
import numpy as np
import cv2
from typing import Any, Tuple
from PIL import Image


class CIFAR10withOOD(dset.CIFAR10):
    def __init__(self, root, batch_size, ood_files=None, download=True, train=True, transform=None,
                 ood_transform=None, use_patch_aug=True, multiclass=True):
        self.root = root
        self.ood_files = ood_files
        self.num_classes = 10
        self.batch_size = batch_size
        self.ood_transform = ood_transform
        self.use_patch_aug = use_patch_aug
        self.multiclass = multiclass
        if multiclass:
            self.ood_prob = (self.batch_size / (self.num_classes + 1) * 0.01)
        else:
            self.ood_prob = 0.5
        super().__init__(root, download=download, train=train, transform=transform)

    def extract_blocks(self, image, mask):
        maximum_score = 8 * 8
        patch_size = 8
        stride = patch_size // 2
        patch_images = []
        mask_score = []
        for i in range(0, 28, stride):
            for j in range(0, 28, stride):
                if (j + patch_size) > 32:
                    break
                patch_image = image[i:i + patch_size, j:j + patch_size]
                patch_score = maximum_score - np.sum(mask[i:i + patch_size, j:j + patch_size])
                if patch_score == maximum_score:
                    if np.random.uniform() >= 0.5:
                        resize_image = cv2.resize(patch_image, (32, 32))
                    else:
                        resize_image = np.tile(patch_image, [4, 4, 1])
                    return resize_image
                patch_images.append(patch_image)
                mask_score.append(patch_score)

            if (i + patch_size) > 32:
                break

        max_idx = np.argmax(mask_score, axis=0)
        max_patch = patch_images[max_idx]
        if np.random.uniform() >= 0.5:
            resize_image = cv2.resize(max_patch, (32, 32))
        else:
            resize_image = np.tile(max_patch, [4, 4, 1])
        return resize_image

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if np.random.uniform() <= self.ood_prob:
            rand_idx = np.random.randint(0, len(self.ood_files))
            img_file = self.ood_files[rand_idx]
            img = np.array(Image.open(img_file).convert('RGB')).astype(np.uint8)
            target = self.num_classes

            if self.use_patch_aug and np.random.uniform() >= 0.5:
                mask_file = img_file.replace("_train.png", "_mask.png")
                mask = np.array(Image.open(mask_file)) / 255.
                mask = np.uint8(mask)
                img = self.extract_blocks(img, mask)

            img = Image.fromarray(img)
            if self.ood_transform is not None:
                img = self.ood_transform(img)

        else:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100withOOD(dset.CIFAR100):
    def __init__(self, root, batch_size, ood_files=None, download=True, train=True, transform=None,
                 ood_transform=None, use_patch_aug=True, multiclass=True):
        self.root = root
        self.ood_files = ood_files
        self.num_classes = 100
        self.batch_size = batch_size
        self.ood_transform = ood_transform
        self.use_patch_aug = use_patch_aug
        self.multiclass = multiclass
        if multiclass:
            self.ood_prob = (self.batch_size / (self.num_classes + 1) * 0.01)
        else:
            self.ood_prob = 0.5
        super().__init__(root, download=download, train=train, transform=transform)

    def extract_blocks(self, image, mask):
        maximum_score = 8 * 8
        patch_size = 8
        stride = patch_size // 2
        patch_images = []
        mask_score = []
        for i in range(0, 32 - stride, stride):
            for j in range(0, 32 - stride, stride):
                if (j + patch_size) > 32:
                    break
                patch_image = image[i:i + patch_size, j:j + patch_size]
                patch_score = maximum_score - np.sum(mask[i:i + patch_size, j:j + patch_size])
                if patch_score == maximum_score:
                    if np.random.uniform() >= 0.5:
                        resize_image = cv2.resize(patch_image, (32, 32))
                    else:
                        resize_image = np.tile(patch_image, [4, 4, 1])
                    return resize_image
                patch_images.append(patch_image)
                mask_score.append(patch_score)

            if (i + patch_size) > 32:
                break

        max_idx = np.argmax(mask_score, axis=0)
        max_patch = patch_images[max_idx]
        if np.random.uniform() >= 0.5:
            resize_image = cv2.resize(max_patch, (32, 32))
        else:
            resize_image = np.tile(max_patch, [4, 4, 1])
        return resize_image

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if np.random.uniform() <= self.ood_prob:
            rand_idx = np.random.randint(0, len(self.ood_files))
            img_file = self.ood_files[rand_idx]
            img = np.array(Image.open(img_file).convert('RGB')).astype(np.uint8)
            target = self.num_classes

            if self.use_patch_aug and np.random.uniform() >= 0.5:
                mask_file = img_file.replace("_train.png", "_mask.png")
                mask = np.array(Image.open(mask_file)) / 255.
                mask = np.uint8(mask)
                img = self.extract_blocks(img, mask)

            img = Image.fromarray(img)
            if self.ood_transform is not None:
                img = self.ood_transform(img)

        else:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target