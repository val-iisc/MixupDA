import os
import os.path as osp
from pathlib import Path
import sys
import random
import torch
import torchvision
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import *
import torchvision.transforms as transforms
import torchvision.transforms.functional as Ft
from styleaug import StyleAugmentor
import imgaug.augmenters as iaa
import cv2
import net
from PIL import ImageFilter



class CityscapesDataset_blur(data.Dataset):
    def __init__(self, label_root, rgb_root, label_path, rgb_path, crop_size, transform=None, limits=None, adain=None, styleaug=None, fda=None, imgaug=None, ignore_label=19):
        '''
        label_root - root path to label images
        rgb_root - root path to RGB images
        label_path - path to list of labels (label_root and paths in this file will be concatenated to get the full path)
        rgb_path - path to list of RGB images (rgb_root and paths in this file will be concatenated to get the full path)
        crop_size - size of images and labels required
        transform - one of ['gamma', 'brightness', 'saturation', 'contrast', 'hue', 'rotate', 'noise', 'bilateral', 'blur', 'rgb_flip', 'gaussian', None]
            (these are the basic augmentations)
        limits - the lower and upper bound of augmentation to be applied (from transform argument), should be a tuple like (0.3, 0.5)
        adain - Adaptive Instance Normalization based style augmentation, the input should be between 0 and 1 and is the 
            degree of stylization.
        styleaug - CVPRW 19 paper's method of stylization, input should be True when required, else don't use the argument
        fda - Fourier DA paper's method of stylization, input should be a tuple 
            First argument should be path to the directory containing style images
            Second argument should be L (or beta) to be used for the stylization
        imgaug - Augmentations using imgaug library, one of ['rain', 'frost', 'snow', 'cartoon']
        '''
        self.ignore_label = ignore_label
        self.label_root = label_root
        self.rgb_root = rgb_root
        self.img_ids = [i_id.strip() for i_id in open(label_path)]
        self.img_rgb_ids = [i_id.strip() for i_id in open(rgb_path)]
        self.crop_size = crop_size
        self.IMG_MEAN = np.asarray((104.00699, 116.66877, 122.67892), np.float32)
        # making mean zero here, since mean subtraction is to be done after FDA conversion
        # self.IMG_MEAN = img_mean
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.transform = transform
        self.limits = limits
        self.imgaug = imgaug
        self.adain = adain
        self.styleaug = styleaug
        self.fda = fda
        self.imgaug = imgaug

        if self.fda:
            self.fda = fda[0]
            self.fda_L = fda[1]
            # fda[0] should have path to directory with target images
            tgt_dir = Path(self.fda)
            self.tgt_paths = [f for f in tgt_dir.glob('*')]

        if self.adain:
            self.vgg = net.vgg
            self.decoder = net.decoder
            self.decoder.eval()
            self.vgg.eval()
            self.decoder.load_state_dict(torch.load('checkpoints/decoder.pth'))
            self.vgg.load_state_dict(torch.load('checkpoints/vgg_normalised.pth'))
            self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
            self.vgg.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.decoder.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.content_tf = test_transform(512, False)
            self.style_tf = test_transform(512, False)

            style_dir = Path('input/style')
            self.style_paths = [f for f in style_dir.glob('*')]

        if self.styleaug:
            self.augmentor = StyleAugmentor()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        rgb = Image.open(os.path.join(self.rgb_root, self.img_rgb_ids[index])).convert('RGB')
        label = Image.open(os.path.join(self.label_root, self.img_ids[index]))

        label = label.resize(self.crop_size, Image.NEAREST)

        label = np.asarray(label, dtype=np.uint8)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)

        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        if self.adain:
            style_choice = random.randint(0, len(self.style_paths) - 1)
            content = self.content_tf(rgb)
            style = self.style_tf(Image.open(str(self.style_paths[style_choice])))
            style = style.to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
            content = content.to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(self.vgg, self.decoder, content, style, alpha=self.adain)
            output = output.cpu().squeeze(0).numpy()
            rgb = toimage(output)

        rgb = rgb.resize(self.crop_size, Image.BICUBIC)

        if self.imgaug:
            rgb = np.asarray(rgb, np.uint8)
            severity = random.randint(1, 3)
            if self.imgaug == 'rain':
                aug = iaa.Rain(drop_size=(0.5, 0.7))
            elif self.imgaug == 'snow':
                aug = iaa.imgcorruptlike.Snow(severity=severity)
            elif self.imgaug == 'frost':
                aug = iaa.imgcorruptlike.Frost(severity=severity)
            elif self.imgaug == 'cartoon':
                aug = iaa.Cartoon()

            rgb = aug(image=rgb)
            rgb = toimage(rgb, channel_axis=2, cmin=0, cmax=255)

        if self.fda:
            source = np.asarray(rgb, np.float32)

            if self.fda == 'random':
                target = np.random.uniform(1, 255, source.shape)
            else:
                choice = random.randint(0, len(self.tgt_paths) - 1)
                target = Image.open(self.tgt_paths[choice])
                target = target.resize(self.crop_size, Image.BICUBIC)

            target = np.asarray(target, np.float32)

            source = source.transpose((2, 0, 1))
            target = target.transpose((2, 0, 1))

            output = FDA_source_to_target_np(source, target, L=self.fda_L)
            rgb = toimage(output, channel_axis=0, cmin=0, cmax=255)

        if self.styleaug:
            imtorch = transforms.ToTensor()(rgb).unsqueeze(0)
            imtorch = imtorch.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                imrestyled = self.augmentor(imtorch)
            imrestyled = imrestyled.cpu().squeeze(0).numpy()
            rgb = toimage(imrestyled)

        blur_rgb = rgb.filter(ImageFilter.BLUR)

        blur_rgb = np.asarray(blur_rgb, np.float32)
        blur_rgb = blur_rgb[:, :, ::-1]  # change to BGR
        blur_rgb -= self.IMG_MEAN
        blur_rgb = blur_rgb.transpose((2, 0, 1)).copy() # (C x H x W)


        rgb = np.asarray(rgb, np.float32)
        rgb = rgb[:, :, ::-1]  # change to BGR
        rgb -= self.IMG_MEAN
        rgb = rgb.transpose((2, 0, 1)).copy() # (C x H x W)
        return label_copy.copy(), rgb.copy(), blur_rgb.copy()

class GTA5Dataset_blur(data.Dataset):
    def __init__(self, root, rgb_root, list_path, crop_size, transform=None, limits=None, adain=None, styleaug=None, fda=None, imgaug=None, mirror=None, ignore_label=19, dataset='gta5'):
        self.root = root
        self.rgb_root = rgb_root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.IMG_MEAN = np.asarray((104.00699, 116.66877, 122.67892), np.float32)
        # making mean zero here, since mean subtraction is to be done after FDA conversion
        # self.IMG_MEAN = np.asarray((0, 0, 0), np.float32)
        self.transform = transform
        self.limits = limits
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.imgaug = imgaug
        self.mirror = mirror
        self.dataset = dataset
        assert self.dataset in ['gta5', 'synscapes']

        if self.dataset == 'gta5':
            self.resize = (1280, 720)
            self.crop_size = (1024, 512)
        else:
            self.resize = (1024, 512)
            self.crop_size = (1024, 512)

        self.fda = fda
        if self.fda:
            self.fda = fda[0]
            self.fda_L = fda[1]
            # fda[0] should have path to directory with target images
            tgt_dir = Path(self.fda)
            self.tgt_paths = [f for f in tgt_dir.glob('*')]

        self.adain = adain
        if self.adain:
            self.vgg = net.vgg
            self.decoder = net.decoder
            self.decoder.eval()
            self.vgg.eval()
            self.decoder.load_state_dict(torch.load('checkpoints/decoder.pth'))
            self.vgg.load_state_dict(torch.load('checkpoints/vgg_normalised.pth'))
            self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
            self.vgg.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.decoder.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.content_tf = test_transform(512, False)
            self.style_tf = test_transform(512, False)

            style_dir = Path('input/style')
            self.style_paths = [f for f in style_dir.glob('*')]

        self.styleaug = styleaug
        if self.styleaug:
            self.augmentor = StyleAugmentor()

        for name in self.img_ids:
            if self.dataset == 'gta5':
                img_file = osp.join(self.rgb_root, "images/%s" % name)
                label_file = osp.join(self.root, "labels/%s" % name)
            else:
                img_file = osp.join(self.rgb_root, "%s" % name)
                label_file = osp.join(self.root, "%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        label = label.resize(self.resize, Image.NEAREST)

        if self.adain:
            style_choice = random.randint(0, len(self.style_paths) - 1)
            content = self.content_tf(image)
            style = self.style_tf(Image.open(str(self.style_paths[style_choice])))
            style = style.to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
            content = content.to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(self.vgg, self.decoder, content, style, alpha=self.adain)
            output = output.cpu().squeeze(0).numpy()
            image = toimage(output)

        image = image.resize(self.resize, Image.BICUBIC)

        if self.imgaug:
            image = np.asarray(image, np.uint8)
            severity = random.randint(1, 3)
            if self.imgaug == 'rain':
                aug = iaa.Rain(drop_size=(0.5, 0.7))
            elif self.imgaug == 'snow':
                aug = iaa.imgcorruptlike.Snow(severity=severity)
            elif self.imgaug == 'frost':
                aug = iaa.imgcorruptlike.Frost(severity=severity)
            elif self.imgaug == 'cartoon':
                aug = iaa.Cartoon()

            image = aug(image=image)
            image = toimage(image, channel_axis=2, cmin=0, cmax=255)

        if self.fda:
            source = np.asarray(image, np.float32)
            if self.fda == 'random':
                target = np.random.uniform(1, 255, source.shape)
            else:
                choice = random.randint(0, len(self.tgt_paths) - 1)
                target = Image.open(self.tgt_paths[choice])
                target = target.resize(self.resize, Image.BICUBIC)

            target = np.asarray(target, np.float32)

            source = source.transpose((2, 0, 1))
            target = target.transpose((2, 0, 1))

            output = FDA_source_to_target_np(source, target, L=self.fda_L)
            image = toimage(output, channel_axis=0, cmin=0, cmax=255)

        if self.styleaug:
            imtorch = transforms.ToTensor()(image).unsqueeze(0)
            imtorch = imtorch.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                imrestyled = self.augmentor(imtorch)
            imrestyled = imrestyled.cpu().squeeze(0).numpy()
            image = toimage(imrestyled)

        if self.transform:
            trad_tf = random.choice([0, 1])
            if trad_tf == 0:
                image = np.asarray(image, np.uint8)
                aug = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
                image = aug(image=image)
                image = toimage(image, channel_axis=2, cmin=0, cmax=255)
            else:
                # average blurring
                sizes = [5, 7, 9]
                k_size = random.randint(0, len(sizes) - 1)
                # kernel = np.ones((int(self.limits[0]), int(self.limits[0])), np.float32) / (int(self.limits[0]) ** 2)
                kernel = np.ones((int(sizes[k_size]), int(sizes[k_size])), np.float32) / (int(sizes[k_size]) ** 2)
                image = np.asarray(image, np.float32)
                image = cv2.filter2D(image, -1, kernel)
                image = toimage(image, channel_axis=2, cmin=0, cmax=255)

        if self.dataset == 'gta5':
            left = self.resize[0]-self.crop_size[0]
            upper= self.resize[1]-self.crop_size[1]

            left = np.random.randint(0, high=left)
            upper= np.random.randint(0, high=upper)
            right= left + self.crop_size[0]
            lower= upper+ self.crop_size[1]

            image = image.crop((left, upper, right, lower))
            label = label.crop((left, upper, right, lower))

        blur_image = image.filter(ImageFilter.BLUR)

        image = np.asarray(image, np.float32)
        blur_image = np.asarray(blur_image, np.float32)
        label = np.asarray(label, np.float32)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)

        # re-assign labels to match the format of Cityscapes
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        image = image[:, :, ::-1]  # change to BGR
        image -= self.IMG_MEAN
        image = image.transpose(2, 0, 1)

        blur_image = blur_image[:, :, ::-1]  # change to BGR
        blur_image -= self.IMG_MEAN
        blur_image = blur_image.transpose(2, 0, 1)

        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            blur_image = blur_image[:, :, ::flip]
            label_copy = label_copy[:, ::flip]

        return label_copy.copy(), image.copy(), blur_image.copy()

class SynthiaDataset_blur(data.Dataset):
    def __init__(self, root, list_path, crop='centre', mirror=None, imgaug=None, fda=None, adain=None, styleaug=None, ignore_label=19):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.IMG_MEAN = np.asarray((104.00699, 116.66877, 122.67892), np.float32)
        self.id_to_trainid = {1:10, 2:2, 3:0, 4:1, 5:4, 6:8, 7:5, 8:13,
                            9:7, 10:11, 11:18, 12:17, 15:6, 16:9, 17:12,
                            18:14, 19:15, 20:16, 21:3}

        self.mirror = mirror
        self.crop = crop

        self.imgaug = imgaug
        self.resize = (1280, 760)
        self.crop_size = (1024, 512)

        self.fda = fda
        if self.fda:
            self.fda = fda[0]
            self.fda_L = fda[1]
            # fda[0] should have path to directory with target images
            tgt_dir = Path(self.fda)
            self.tgt_paths = [f for f in tgt_dir.glob('*')]

        self.adain = adain
        if self.adain:
            self.vgg = net.vgg
            self.decoder = net.decoder
            self.decoder.eval()
            self.vgg.eval()
            self.decoder.load_state_dict(torch.load('checkpoints/decoder.pth'))
            self.vgg.load_state_dict(torch.load('checkpoints/vgg_normalised.pth'))
            self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
            self.vgg.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.decoder.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.content_tf = test_transform(512, False)
            self.style_tf = test_transform(512, False)

            style_dir = Path('input/style')
            self.style_paths = [f for f in style_dir.glob('*')]

        self.styleaug = styleaug
        if self.styleaug:
            self.augmentor = StyleAugmentor()

        for name in self.img_ids:
            img_file = osp.join(self.root, "RGB/" + "{:0>7d}.png".format(int(name)))
            label_file = osp.join(self.root, "GT/parsed_LABELS/" + "{:0>7d}.png".format(int(name)))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        label = label.resize(self.resize, Image.NEAREST)

        if self.adain:
            style_choice = random.randint(0, len(self.style_paths) - 1)
            content = self.content_tf(image)
            style = self.style_tf(Image.open(str(self.style_paths[style_choice])))
            style = style.to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
            content = content.to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(self.vgg, self.decoder, content, style, alpha=self.adain)
            output = output.cpu().squeeze(0).numpy()
            image = toimage(output)

        image = image.resize(self.resize, Image.BICUBIC)

        if self.imgaug:
            image = np.asarray(image, np.uint8)
            severity = random.randint(1, 3)
            if self.imgaug == 'snow':
                aug = iaa.imgcorruptlike.Snow(severity=severity)
            elif self.imgaug == 'frost':
                aug = iaa.imgcorruptlike.Frost(severity=severity)
            elif self.imgaug == 'cartoon':
                aug = iaa.Cartoon()

            image = aug(image=image)
            image = toimage(image, channel_axis=2, cmin=0, cmax=255)

        if self.fda:
            source = np.asarray(image, np.float32)
            if self.fda == 'random':
                target = np.random.uniform(1, 255, source.shape)
            else:
                choice = random.randint(0, len(self.tgt_paths) - 1)
                target = Image.open(self.tgt_paths[choice])
                target = target.resize(self.resize, Image.BICUBIC)

            target = np.asarray(target, np.float32)

            source = source.transpose((2, 0, 1))
            target = target.transpose((2, 0, 1))

            output = FDA_source_to_target_np(source, target, L=self.fda_L)
            image = toimage(output, channel_axis=0, cmin=0, cmax=255)

        if self.styleaug:
            imtorch = transforms.ToTensor()(image).unsqueeze(0)
            imtorch = imtorch.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                imrestyled = self.augmentor(imtorch)
            imrestyled = imrestyled.cpu().squeeze(0).numpy()
            image = toimage(imrestyled)

        if self.crop == 'random':
            # random cropping from self.resize to self.crop_size
            left = self.resize[0]-self.crop_size[0]
            upper= self.resize[1]-self.crop_size[1]

            left = np.random.randint(0, high=left)
            upper= np.random.randint(0, high=upper)
            right= left + self.crop_size[0]
            lower= upper+ self.crop_size[1]
        elif self.crop == 'centre':
            ### For centre cropping
            left = (self.resize[0] - self.crop_size[0]) // 2
            upper = (self.resize[1] - self.crop_size[1]) // 2
            right = left + self.crop_size[0]
            lower = upper + self.crop_size[1]

        image = image.crop((left, upper, right, lower))
        label = label.crop((left, upper, right, lower))

        blur_image = image.filter(ImageFilter.BLUR)

        image = np.asarray(image, np.float32)
        blur_image = np.asarray(blur_image, np.float32)

        label = np.asarray(label, dtype=np.uint8)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)

        # re-assign labels to match the format of Cityscapes
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        image = image[:, :, ::-1]  # change to BGR
        image -= self.IMG_MEAN
        image = image.transpose(2, 0, 1)

        blur_image = blur_image[:, :, ::-1]  # change to BGR
        blur_image -= self.IMG_MEAN
        blur_image = blur_image.transpose(2, 0, 1)

        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            blur_image = blur_image[:, :, ::flip]
            label_copy = label_copy[:, ::flip]

        return label_copy.copy(), image.copy(), blur_image.copy()

class CityscapesDataset(data.Dataset):
    def __init__(self, label_root, rgb_root, label_path, rgb_path, crop_size, transform=None, limits=None, adain=None, styleaug=None, fda=None, imgaug=None, ignore_label=19):
        '''
        label_root - root path to label images
        rgb_root - root path to RGB images
        label_path - path to list of labels (label_root and paths in this file will be concatenated to get the full path)
        rgb_path - path to list of RGB images (rgb_root and paths in this file will be concatenated to get the full path)
        crop_size - size of images and labels required
        transform - one of ['gamma', 'brightness', 'saturation', 'contrast', 'hue', 'rotate', 'noise', 'bilateral', 'blur', 'rgb_flip', 'gaussian', None]
            (these are the basic augmentations)
        limits - the lower and upper bound of augmentation to be applied (from transform argument), should be a tuple like (0.3, 0.5)
        adain - Adaptive Instance Normalization based style augmentation, the input should be between 0 and 1 and is the 
            degree of stylization.
        styleaug - CVPRW 19 paper's method of stylization, input should be True when required, else don't use the argument
        fda - Fourier DA paper's method of stylization, input should be a tuple 
            First argument should be path to the directory containing style images
            Second argument should be L (or beta) to be used for the stylization
        imgaug - Augmentations using imgaug library, one of ['rain', 'frost', 'snow', 'cartoon']
        '''
        self.ignore_label = ignore_label
        self.label_root = label_root
        self.rgb_root = rgb_root
        self.img_ids = [i_id.strip() for i_id in open(label_path)]
        self.img_rgb_ids = [i_id.strip() for i_id in open(rgb_path)]
        self.crop_size = crop_size
        self.IMG_MEAN = np.asarray((104.00699, 116.66877, 122.67892), np.float32)
        # making mean zero here, since mean subtraction is to be done after FDA conversion
        # self.IMG_MEAN = img_mean
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.transform = transform
        self.limits = limits
        self.imgaug = imgaug
        self.adain = adain
        self.styleaug = styleaug
        self.fda = fda
        self.imgaug = imgaug

        if self.fda:
            self.fda = fda[0]
            self.fda_L = fda[1]
            # fda[0] should have path to directory with target images
            tgt_dir = Path(self.fda)
            self.tgt_paths = [f for f in tgt_dir.glob('*')]

        if self.adain:
            self.vgg = net.vgg
            self.decoder = net.decoder
            self.decoder.eval()
            self.vgg.eval()
            self.decoder.load_state_dict(torch.load('checkpoints/decoder.pth'))
            self.vgg.load_state_dict(torch.load('checkpoints/vgg_normalised.pth'))
            self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
            self.vgg.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.decoder.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.content_tf = test_transform(512, False)
            self.style_tf = test_transform(512, False)

            style_dir = Path('input/style')
            self.style_paths = [f for f in style_dir.glob('*')]

        if self.styleaug:
            self.augmentor = StyleAugmentor()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        rgb = Image.open(os.path.join(self.rgb_root, self.img_rgb_ids[index])).convert('RGB')
        label = Image.open(os.path.join(self.label_root, self.img_ids[index]))

        label = label.resize(self.crop_size, Image.NEAREST)

        label = np.asarray(label, dtype=np.uint8)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)

        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        if self.adain:
            style_choice = random.randint(0, len(self.style_paths) - 1)
            content = self.content_tf(rgb)
            style = self.style_tf(Image.open(str(self.style_paths[style_choice])))
            style = style.to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
            content = content.to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(self.vgg, self.decoder, content, style, alpha=self.adain)
            output = output.cpu().squeeze(0).numpy()
            rgb = toimage(output)

        rgb = rgb.resize(self.crop_size, Image.BICUBIC)

        if self.imgaug:
            rgb = np.asarray(rgb, np.uint8)
            severity = random.randint(1, 3)
            if self.imgaug == 'rain':
                aug = iaa.Rain(drop_size=(0.5, 0.7))
            elif self.imgaug == 'snow':
                aug = iaa.imgcorruptlike.Snow(severity=severity)
            elif self.imgaug == 'frost':
                aug = iaa.imgcorruptlike.Frost(severity=severity)
            elif self.imgaug == 'cartoon':
                aug = iaa.Cartoon()

            rgb = aug(image=rgb)
            rgb = toimage(rgb, channel_axis=2, cmin=0, cmax=255)

        if self.fda:
            source = np.asarray(rgb, np.float32)

            if self.fda == 'random':
                target = np.random.uniform(1, 255, source.shape)
            else:
                choice = random.randint(0, len(self.tgt_paths) - 1)
                target = Image.open(self.tgt_paths[choice])
                target = target.resize(self.crop_size, Image.BICUBIC)

            target = np.asarray(target, np.float32)

            source = source.transpose((2, 0, 1))
            target = target.transpose((2, 0, 1))

            output = FDA_source_to_target_np(source, target, L=self.fda_L)
            rgb = toimage(output, channel_axis=0, cmin=0, cmax=255)

        if self.styleaug:
            imtorch = transforms.ToTensor()(rgb).unsqueeze(0)
            imtorch = imtorch.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                imrestyled = self.augmentor(imtorch)
            imrestyled = imrestyled.cpu().squeeze(0).numpy()
            rgb = toimage(imrestyled)

        rgb = np.asarray(rgb, np.float32)
        rgb = rgb[:, :, ::-1]  # change to BGR
        rgb -= self.IMG_MEAN
        rgb = rgb.transpose((2, 0, 1)).copy() # (C x H x W)
        return label_copy.copy(), rgb.copy()

class GTA5Dataset(data.Dataset):
    def __init__(self, root, rgb_root, list_path, crop_size, crop = 'centre', transform=None, limits=None, adain=None, styleaug=None, fda=None, imgaug=None, mirror=None, ignore_label=19, dataset='gta5'):
        self.root = root
        self.rgb_root = rgb_root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.IMG_MEAN = np.asarray((104.00699, 116.66877, 122.67892), np.float32)
        # making mean zero here, since mean subtraction is to be done after FDA conversion
        # self.IMG_MEAN = np.asarray((0, 0, 0), np.float32)
        self.transform = transform
        self.limits = limits
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        self.imgaug = imgaug
        self.mirror = mirror
        self.dataset = dataset
        self.crop = crop
        assert self.dataset in ['gta5', 'synscapes']

        if self.dataset == 'gta5':
            self.resize = (1280, 720)
            self.crop_size = (1024, 512)
        else:
            self.resize = (1024, 512)
            self.crop_size = (1024, 512)

        self.fda = fda
        if self.fda:
            self.fda = fda[0]
            self.fda_L = fda[1]
            # fda[0] should have path to directory with target images
            tgt_dir = Path(self.fda)
            self.tgt_paths = [f for f in tgt_dir.glob('*')]

        self.adain = adain
        if self.adain:
            self.vgg = net.vgg
            self.decoder = net.decoder
            self.decoder.eval()
            self.vgg.eval()
            self.decoder.load_state_dict(torch.load('checkpoints/decoder.pth'))
            self.vgg.load_state_dict(torch.load('checkpoints/vgg_normalised.pth'))
            self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
            self.vgg.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.decoder.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.content_tf = test_transform(512, False)
            self.style_tf = test_transform(512, False)

            style_dir = Path('input/style')
            self.style_paths = [f for f in style_dir.glob('*')]

        self.styleaug = styleaug
        if self.styleaug:
            self.augmentor = StyleAugmentor()

        for name in self.img_ids:
            if self.dataset == 'gta5':
                img_file = osp.join(self.rgb_root, "images/%s" % name)
                label_file = osp.join(self.root, "labels/%s" % name)
            else:
                img_file = osp.join(self.rgb_root, "%s" % name)
                label_file = osp.join(self.root, "%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        label = label.resize(self.resize, Image.NEAREST)

        if self.adain:
            style_choice = random.randint(0, len(self.style_paths) - 1)
            content = self.content_tf(image)
            style = self.style_tf(Image.open(str(self.style_paths[style_choice])))
            style = style.to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
            content = content.to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(self.vgg, self.decoder, content, style, alpha=self.adain)
            output = output.cpu().squeeze(0).numpy()
            image = toimage(output)

        image = image.resize(self.resize, Image.BICUBIC)

        if self.imgaug:
            image = np.asarray(image, np.uint8)
            severity = random.randint(1, 3)
            if self.imgaug == 'rain':
                aug = iaa.Rain(drop_size=(0.5, 0.7))
            elif self.imgaug == 'snow':
                aug = iaa.imgcorruptlike.Snow(severity=severity)
            elif self.imgaug == 'frost':
                aug = iaa.imgcorruptlike.Frost(severity=severity)
            elif self.imgaug == 'cartoon':
                aug = iaa.Cartoon()

            image = aug(image=image)
            image = toimage(image, channel_axis=2, cmin=0, cmax=255)

        if self.fda:
            source = np.asarray(image, np.float32)
            if self.fda == 'random':
                target = np.random.uniform(1, 255, source.shape)
            else:
                choice = random.randint(0, len(self.tgt_paths) - 1)
                target = Image.open(self.tgt_paths[choice])
                target = target.resize(self.resize, Image.BICUBIC)

            target = np.asarray(target, np.float32)

            source = source.transpose((2, 0, 1))
            target = target.transpose((2, 0, 1))

            output = FDA_source_to_target_np(source, target, L=self.fda_L)
            image = toimage(output, channel_axis=0, cmin=0, cmax=255)

        if self.styleaug:
            imtorch = transforms.ToTensor()(image).unsqueeze(0)
            imtorch = imtorch.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                imrestyled = self.augmentor(imtorch)
            imrestyled = imrestyled.cpu().squeeze(0).numpy()
            image = toimage(imrestyled)

        if self.transform:
            trad_tf = random.choice([0, 1])
            if trad_tf == 0:
                image = np.asarray(image, np.uint8)
                aug = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
                image = aug(image=image)
                image = toimage(image, channel_axis=2, cmin=0, cmax=255)
            else:
                # average blurring
                sizes = [5, 7, 9]
                k_size = random.randint(0, len(sizes) - 1)
                # kernel = np.ones((int(self.limits[0]), int(self.limits[0])), np.float32) / (int(self.limits[0]) ** 2)
                kernel = np.ones((int(sizes[k_size]), int(sizes[k_size])), np.float32) / (int(sizes[k_size]) ** 2)
                image = np.asarray(image, np.float32)
                image = cv2.filter2D(image, -1, kernel)
                image = toimage(image, channel_axis=2, cmin=0, cmax=255)

        if self.dataset == 'gta5':
            if self.crop == 'random':
                # random cropping from self.resize to self.crop_size
                left = self.resize[0]-self.crop_size[0]
                upper= self.resize[1]-self.crop_size[1]

                left = np.random.randint(0, high=left)
                upper= np.random.randint(0, high=upper)
                right= left + self.crop_size[0]
                lower= upper+ self.crop_size[1]
            elif self.crop == 'centre':
                ### For centre cropping
                left = (self.resize[0] - self.crop_size[0]) // 2
                upper = (self.resize[1] - self.crop_size[1]) // 2
                right = left + self.crop_size[0]
                lower = upper + self.crop_size[1]

            image = image.crop((left, upper, right, lower))
            label = label.crop((left, upper, right, lower))

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)

        # re-assign labels to match the format of Cityscapes
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        image = image[:, :, ::-1]  # change to BGR
        image -= self.IMG_MEAN
        image = image.transpose(2, 0, 1)

        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label_copy = label_copy[:, ::flip]

        return label_copy.copy(), image.copy()

class SynthiaDataset(data.Dataset):
    def __init__(self, root, list_path, crop='centre', mirror=None, imgaug=None, fda=None, adain=None, styleaug=None, ignore_label=19):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.IMG_MEAN = np.asarray((104.00699, 116.66877, 122.67892), np.float32)
        self.id_to_trainid = {1:10, 2:2, 3:0, 4:1, 5:4, 6:8, 7:5, 8:13,
                            9:7, 10:11, 11:18, 12:17, 15:6, 16:9, 17:12,
                            18:14, 19:15, 20:16, 21:3}

        self.mirror = mirror
        self.crop = crop

        self.imgaug = imgaug
        self.resize = (1280, 760)
        self.crop_size = (1024, 512)

        self.fda = fda
        if self.fda:
            self.fda = fda[0]
            self.fda_L = fda[1]
            # fda[0] should have path to directory with target images
            tgt_dir = Path(self.fda)
            self.tgt_paths = [f for f in tgt_dir.glob('*')]

        self.adain = adain
        if self.adain:
            self.vgg = net.vgg
            self.decoder = net.decoder
            self.decoder.eval()
            self.vgg.eval()
            self.decoder.load_state_dict(torch.load('checkpoints/decoder.pth'))
            self.vgg.load_state_dict(torch.load('checkpoints/vgg_normalised.pth'))
            self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
            self.vgg.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.decoder.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.content_tf = test_transform(512, False)
            self.style_tf = test_transform(512, False)

            style_dir = Path('input/style')
            self.style_paths = [f for f in style_dir.glob('*')]

        self.styleaug = styleaug
        if self.styleaug:
            self.augmentor = StyleAugmentor()

        for name in self.img_ids:
            img_file = osp.join(self.root, "RGB/" + "{:0>7d}.png".format(int(name)))
            label_file = osp.join(self.root, "GT/parsed_LABELS/" + "{:0>7d}.png".format(int(name)))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        label = label.resize(self.resize, Image.NEAREST)

        if self.adain:
            style_choice = random.randint(0, len(self.style_paths) - 1)
            content = self.content_tf(image)
            style = self.style_tf(Image.open(str(self.style_paths[style_choice])))
            style = style.to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
            content = content.to('cuda:0' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(self.vgg, self.decoder, content, style, alpha=self.adain)
            output = output.cpu().squeeze(0).numpy()
            image = toimage(output)

        image = image.resize(self.resize, Image.BICUBIC)

        if self.imgaug:
            image = np.asarray(image, np.uint8)
            severity = random.randint(1, 3)
            if self.imgaug == 'snow':
                aug = iaa.imgcorruptlike.Snow(severity=severity)
            elif self.imgaug == 'frost':
                aug = iaa.imgcorruptlike.Frost(severity=severity)
            elif self.imgaug == 'cartoon':
                aug = iaa.Cartoon()

            image = aug(image=image)
            image = toimage(image, channel_axis=2, cmin=0, cmax=255)

        if self.fda:
            source = np.asarray(image, np.float32)
            if self.fda == 'random':
                target = np.random.uniform(1, 255, source.shape)
            else:
                choice = random.randint(0, len(self.tgt_paths) - 1)
                target = Image.open(self.tgt_paths[choice])
                target = target.resize(self.resize, Image.BICUBIC)

            target = np.asarray(target, np.float32)

            source = source.transpose((2, 0, 1))
            target = target.transpose((2, 0, 1))

            output = FDA_source_to_target_np(source, target, L=self.fda_L)
            image = toimage(output, channel_axis=0, cmin=0, cmax=255)

        if self.styleaug:
            imtorch = transforms.ToTensor()(image).unsqueeze(0)
            imtorch = imtorch.to('cuda:0' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                imrestyled = self.augmentor(imtorch)
            imrestyled = imrestyled.cpu().squeeze(0).numpy()
            image = toimage(imrestyled)

        if self.crop == 'random':
            # random cropping from self.resize to self.crop_size
            left = self.resize[0]-self.crop_size[0]
            upper= self.resize[1]-self.crop_size[1]

            left = np.random.randint(0, high=left)
            upper= np.random.randint(0, high=upper)
            right= left + self.crop_size[0]
            lower= upper+ self.crop_size[1]
        elif self.crop == 'centre':
            ### For centre cropping
            left = (self.resize[0] - self.crop_size[0]) // 2
            upper = (self.resize[1] - self.crop_size[1]) // 2
            right = left + self.crop_size[0]
            lower = upper + self.crop_size[1]

        image = image.crop((left, upper, right, lower))
        label = label.crop((left, upper, right, lower))

        image = np.asarray(image, np.float32)

        label = np.asarray(label, dtype=np.uint8)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)

        # re-assign labels to match the format of Cityscapes
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        image = image[:, :, ::-1]  # change to BGR
        image -= self.IMG_MEAN
        image = image.transpose(2, 0, 1)

        if self.mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label_copy = label_copy[:, ::flip]

        return label_copy.copy(), image.copy()

'''
transform = transforms.Compose([
        transforms.TenCrop((int(400 * 0.8), int(800 * 0.8))),
    ])
# dataset = CityscapesDataset(label_root='/sdc1/datasets/cityscape/gtFine',
#                             rgb_root='/sdc1/datasets/cityscape/leftImg8bit',
#                             label_path='cityscapes_random_train.txt',
#                             rgb_path='rgb_cityscapes_random_train.txt',
#                             crop_size=(800, 400),
#                             transform=transform)
dataset = CityscapesDataset(label_root='/sdc1/datasets/gta5-dataset/labels',
                            rgb_root='/sdc1/datasets/gta5-dataset/images',
                            label_path='gta5_random.txt',
                            rgb_path='gta5_random.txt',
                            crop_size=(800, 400),
                            transform=transform)
trainloader = data.DataLoader(dataset, batch_size=4, shuffle=True)

if __name__ == '__main__':
    for i, data in enumerate(trainloader):
        labels, rgbs = data
        labels = labels.reshape(-1, 1, labels.shape[2], labels.shape[3])
        rgbs = rgbs.reshape(-1, rgbs.shape[2], rgbs.shape[3], rgbs.shape[4])
        print(labels.shape)
        print(rgbs.shape)
        if i == 0:
            img = np.transpose(labels[0].numpy(), (1, 2, 0))
            img = label2Color(img)

            image = np.transpose(np.asarray(rgbs[0].squeeze(0), dtype=np.float32), (1, 2, 0))
            IMG_MEAN = np.asarray((104.00699, 116.66877, 122.67892), np.float32)
            image += IMG_MEAN
            image = image[:, :, : : -1]
            image = image.astype(np.uint8)
            plt.subplot(211)
            plt.imshow(image)
            plt.subplot(212)
            plt.imshow(img)
            plt.savefig('label.png')
            sys.exit()

dataset = CityscapesDataset(label_root='/sdc1/datasets/cityscape/gtFine',
                            rgb_root='/sdc1/datasets/cityscape/leftImg8bit',
                            label_path='cityscapes_random_val.txt',
                            rgb_path='rgb_cityscapes_random_val.txt',
                            crop_size=(640, 320),
                            transform=None)
trainloader = data.DataLoader(dataset, batch_size=1, shuffle=True)

if __name__ == '__main__':
    for i, data in enumerate(trainloader):
        labels, rgbs = data
        # labels = labels.reshape(-1, 1, labels.shape[2], labels.shape[3])
        # rgbs = rgbs.reshape(-1, rgbs.shape[2], rgbs.shape[3], rgbs.shape[4])
        print(labels.shape)
        print(rgbs.shape)
        if i == 0:
            img = np.transpose(labels.numpy(), (1, 2, 0))
            img = label2Color(img)

            image = np.transpose(np.asarray(rgbs[0], dtype=np.float32), (1, 2, 0))
            IMG_MEAN = np.asarray((104.00699, 116.66877, 122.67892), np.float32)
            image += IMG_MEAN
            image = image[:, :, : : -1]
            image = image.astype(np.uint8)
            plt.subplot(211)
            plt.imshow(image)
            plt.subplot(212)
            plt.imshow(img)
            plt.savefig('label.png')
            sys.exit()

dataset = SynthiaDataset(root='/sdd1/amit/dataset/synthia_cityscape/RAND_CITYSCAPES',
                         list_path='synthia_test.txt',
                         crop_size=(1280, 720),
                         transform=None)
trainloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
if __name__ == '__main__':
    for i, data in enumerate(trainloader):
        labels, rgbs = data
        # labels = labels.reshape(-1, 1, labels.shape[2], labels.shape[3])
        # rgbs = rgbs.reshape(-1, rgbs.shape[2], rgbs.shape[3], rgbs.shape[4])
        print(labels.shape)
        print(rgbs.shape)
        if i < 10:
            img = np.transpose(labels.numpy(), (1, 2, 0))
            img = label2Color(img)

            image = np.transpose(np.asarray(rgbs[0], dtype=np.float32), (1, 2, 0))
            IMG_MEAN = np.asarray((104.00699, 116.66877, 122.67892), np.float32)
            image += IMG_MEAN
            image = image[:, :, : : -1]
            image = image.astype(np.uint8)
            plt.axis('off')
            plt.subplot(211)
            plt.imshow(image)
            plt.subplot(212)
            plt.imshow(img)
            plt.savefig('synthia_testing/label' + str(i) + '.png', transparent=True)
            plt.clf()
        else:
            sys.exit()
'''

'''
if __name__ == '__main__':
    base = '/data/akshay/machine54'
    root_path = '/sdd1/intern/akshay/deeplab_augment/'
    list_path = 'gta5_val20.txt'
    # dataset = GTA5Dataset(
        # root=base+'/sdc1/datasets/gta5-dataset',
        # list_path='gta5_val20.txt',
        # crop_size=(1280, 720),
        # transform='gaussian',
        # limits=(30, 30),
        # adain=0.3,
        # styleaug=True,
        # fda=('input/idd', 0.005),
        # imgaug='snow'
    # )
    from aug_data import get_datasets
    dataset_list = get_datasets(base, root_path, list_path)
    # dataset = CityscapesDataset(label_root=base+'/sdc1/datasets/cityscape/gtFine',
                                # rgb_root=base+'/sdc1/datasets/cityscape/leftImg8bit',
                                # label_path='cityscapes_val.txt',
                                # rgb_path='rgb_cityscapes_val.txt',
                                # crop_size=(1280, 720),
                                # transform='gaussian',
                                # adain=0.3,
                                # styleaug=True,
                                # fda=('style', 0.005),
                                # imgaug='cartoon'
                                # )
    # trainloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    loader_list = list()
    for dataset in dataset_list:
        loader_list.append(data.DataLoader(dataset, batch_size=4, shuffle=True))
    iter_list = list()
    for loader in loader_list:
        iter_list.append(iter(loader))

    # for i, data in enumerate(trainloader):
    for i in range(20000):
        k = random.randint(0, len(loader_list) - 1)
        try:
            labels, rgbs = iter_list[k].next()
        except StopIteration:
            print('exception encountered at k = ', k)
            iter_list[k] = iter(loader_list[k])
            labels, rgbs = iter_list[k].next()
        # labels, rgbs = data
        # labels = labels.reshape(-1, 1, labels.shape[2], labels.shape[3])
        # rgbs = rgbs.reshape(-1, rgbs.shape[2], rgbs.shape[3], rgbs.shape[4])
        # print(labels.shape)
        # print(rgbs.shape)
        if i < 20:
            img = np.transpose(labels[0].unsqueeze(0).numpy(), (1, 2, 0))
            img = label2Color(img)

            image = np.transpose(np.asarray(rgbs[0], dtype=np.float32), (1, 2, 0))
            IMG_MEAN = np.asarray((104.00699, 116.66877, 122.67892), np.float32)
            image += IMG_MEAN
            image = image[:, :, : : -1]
            image = image.astype(np.uint8)
            both = np.concatenate([image, img], axis=1)
            plt.imsave('testing/combined_' + str(i) + '.png', both)
        else:
            if i % 500 == 0:
                print(i)
            # sys.exit()
'''

##############################################################################################

'''
if __name__ == '__main__':
    # base = '/home/cds/akshay/machine54'
    base=''

    # dataset = GTA5Dataset(
        # root=base+'/sdc1/datasets/gta5-dataset',
        # rgb_root=base+'/sdc1/datasets/gta5-dataset',
        # list_path='gta5_val20.txt',
        # crop_size=(1280, 720),
        # adain=0.3,
        # styleaug=True,
        # fda=('input/idd', 0.005),
        # imgaug='snow'
        # transform=True
    # )

    # dataset = CityscapesDataset(label_root=base+'/sdc1/datasets/cityscape/gtFine',
                                # rgb_root=base+'/sdc1/datasets/cityscape/leftImg8bit',
                                # label_path='cityscapes_val.txt',
                                # rgb_path='rgb_cityscapes_val.txt',
                                # crop_size=(1280, 720),
                                # transform='gaussian',
                                # adain=0.3,
                                # styleaug=True,
                                # fda=('random', 0.01),
                                # imgaug='cartoon'
                                # )
    dataset = SynthiaDataset(
        root=base+'/sdd1/amit/dataset/synthia_cityscape/RAND_CITYSCAPES',
        list_path='synthia_train.txt',
        mirror=True,
        imgaug='frost'
    )

    trainloader = data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(trainloader):
        labels, rgbs = data
        print(labels.shape, rgbs.shape)
        # labels = labels.reshape(-1, 1, labels.shape[2], labels.shape[3])
        # rgbs = rgbs.reshape(-1, rgbs.shape[2], rgbs.shape[3], rgbs.shape[4])
        if i < 20:
            img = np.transpose(labels.numpy(), (1, 2, 0))
            img = label2Color(img)

            image = np.transpose(np.asarray(rgbs[0], dtype=np.float32), (1, 2, 0))
            IMG_MEAN = np.asarray((104.00699, 116.66877, 122.67892), np.float32)
            image += IMG_MEAN
            image = image[:, :, : : -1]
            image = image.astype(np.uint8)
            both = np.concatenate([image, img], axis=1)
            plt.imsave('synthia_testing/frost' + str(i) + '.png', both)
        else:
            # if i % 500 == 0:
                # print(i)
            sys.exit()
'''
