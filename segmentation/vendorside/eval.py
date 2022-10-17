import time
import argparse
from dataset import GTA5Dataset, CityscapesDataset
import torch
import sys
import torchvision
import numpy as np
import torch.optim as optim
import os
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from utils import *
import torch.autograd as autograd
# from deeplab_multi import DeeplabMulti
from deeplab import Res_Deeplab
from fcn8s import VGG16_FCN8s
from torchvision import transforms
from metrics import StreamSegMetrics
from aug_data import get_datasets, get_city_datasets


def eval(FLAGS):
    # CUDA_VISIBLE_DEVICES=0 python main.py
    device = torch.device('cpu') # set the device to cpu
    if(torch.cuda.is_available()): # check if cuda is available
        device = torch.device('cuda:0') # if cuda, set device to cuda
    torch.cuda.empty_cache()

    if FLAGS.visualize:
        writer_name = FLAGS.dataset + '_' + FLAGS.load_model
        writer = SummaryWriter('runs/' + writer_name)
        print('visualizing at runs/', writer_name)

    # gta_model = DeeplabMulti(num_classes=19, pretrained=False).to(device)
    if FLAGS.model == 'deeplab':
        gta_model = Res_Deeplab(num_classes=19, pretrained=False).to(device)
    elif FLAGS.model == 'fcn':
        gta_model = VGG16_FCN8s(num_classes=19).to(device)

    print('loading model from ', FLAGS.load_model)
    # gta_model.load_state_dict(torch.load(FLAGS.load_prev_model))
    gta_model.load_state_dict(torch.load(FLAGS.load_model))
    gta_model.eval()
    print('gta model loading done')

    if FLAGS.dataset == 'gta':
        valid_dst = GTA5Dataset(
            root = FLAGS.base+'/datasets/gta5-dataset',
            list_path='gta5_val20.txt',
            crop_size=(1280, 720),
            transform=None)
    elif FLAGS.dataset == 'cityscapes':
        valid_dst = CityscapesDataset(
            label_root=FLAGS.base+'/datasets/cityscape/gtFine',
            rgb_root=FLAGS.base+'/datasets/cityscape/leftImg8bit',
            label_path='cityscapes_val.txt',
            rgb_path='rgb_cityscapes_val.txt',
            crop_size=(1024, 512),
        )

    valloader = data.DataLoader(valid_dst, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    val_metrics = StreamSegMetrics(19)

    def validate(loader):
        add_img = 100
        val_metrics.reset()
        with torch.no_grad():
            for i, (batch, rgb_batch) in enumerate(loader):
                rgb_batch = rgb_batch.to(device=device, dtype=torch.float)
                batch = batch.to(device=device, dtype=torch.int64)

                input_size = rgb_batch.size()[2:]
                pred = gta_model(rgb_batch)
                pred = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)

                preds = pred.detach().max(dim=1)[1].cpu().numpy()
                targets = batch.cpu().numpy()

                val_metrics.update(targets, preds)

                if i % add_img == 0 and FLAGS.visualize:
                    grd = segMap3(rgb_batch, batch, pred)
                    writer.add_image('eval/rgb_label_pred', grd, i)
                    print(i)
                elif i % add_img == 0:
                    print(i)

            score = val_metrics.get_results()
        return score

    score = validate(valloader)
    print('16 miou on ', FLAGS.dataset, ' : ', score['Sixteen IoU'])
    print('13 miou on ', FLAGS.dataset, ' : ', score['Thirteen IoU'])
    print('19 miou on ', FLAGS.dataset, ' : ', score['Mean IoU'])
    print('class wise IoUs: ', score['Class IoU'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, choices=['deeplab', 'fcn'], default='deeplab')
    parser.add_argument('--dataset', type=str, choices=['cityscapes', 'gta5'], default='cityscapes', help='which dataset to eval on')
    parser.add_argument('--load_model', type=str, help='path to model weights to be evaluated')
    parser.add_argument('--visualize', type=bool, default=False, help='whether to visualize eval')
    parser.add_argument('--base', type=str, default='')
    parser.add_argument('--root_path', type=str, default='/path/to/folder/with/code/')
    FLAGS = parser.parse_args()
    eval(FLAGS)
