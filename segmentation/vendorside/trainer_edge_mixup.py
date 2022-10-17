import random
from dataset import CityscapesDataset_blur
import torch
import torchvision
import numpy as np
import torch.optim as optim
import os
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# from apex import amp
from torch.cuda.amp import autocast, GradScaler
from itertools import cycle
from utils import *
import torch.autograd as autograd
from deeplab import Res_Deeplab
from fcn8s import VGG16_FCN8s
from aug_data import get_gta5_datasets_blur, get_synthia_datasets_blur, get_synscapes_datasets_blur
from torchvision import transforms
from metrics import StreamSegMetrics
from lovasz_losses import lovasz_softmax
from edge_model import DexiNet


def trainingProcedure(FLAGS):
    # CUDA_VISIBLE_DEVICES=0 python main.py
    device = torch.device('cpu') # set the device to cpu
    if(torch.cuda.is_available()): # check if cuda is available
        device = torch.device('cuda:0') # if cuda, set device to cuda
    torch.cuda.empty_cache()
    writer = SummaryWriter('./runs/' + FLAGS.runs)

    if FLAGS.model == 'deeplab':
        gta_model = Res_Deeplab(num_classes=19).to(device)
    elif FLAGS.model == 'fcn8s':
        gta_model = VGG16_FCN8s(num_classes=19).to(device)

    city_val_dst = CityscapesDataset_blur(
        label_root='./datasets/cityscape/gtFine',
        rgb_root='./datasets/cityscape/leftImg8bit',
        label_path='./datasets/cityscapes_val.txt',
        rgb_path='./datasets/rgb_cityscapes_val.txt',
        crop_size=(1024, 512),
        transform=None
    )

    # get multiple datasets with augmentations
    if 'gta5' in FLAGS.dataset:
        gta5_dataset_list = get_gta5_datasets_blur(list_path='./datasets/gta5_train20.txt')
    if 'synthia' in FLAGS.dataset:
        synthia_dataset_list = get_synthia_datasets_blur(list_path='./datasets/synthia_train.txt', crop='random', mirror=True)
    if 'synscapes' in FLAGS.dataset:
        synscapes_dataset_list = get_synscapes_datasets_blur(list_path='./datasets/synscapes_train.txt', mirror=True)

    if FLAGS.model == 'deeplab':
        params = list(gta_model.parameters())

        optimizer = optim.SGD(params, lr=FLAGS.lr, momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iteration: (1 - iteration / FLAGS.end_iter) ** 0.9)
    else:
        # only implemented for deeplab
        return -1

    scaler = GradScaler()

    if FLAGS.load_saved:
        print('loading model from ', FLAGS.load_prev_model)
        gta_model.load_state_dict(torch.load('./checkpoints/' + FLAGS.load_prev_model))
    gta_model.train()

    edge_model = DexiNet().to(device)
    
    print("loading Edge model checkpoint from : ", './checkpoints/24_model.pth')
    edge_model.load_state_dict(torch.load('./checkpoints/24_model.pth', map_location=device))

    # Put model in evaluation mode
    edge_model.eval()
    
    print('model loading done')

    cityvalloader = data.DataLoader(city_val_dst, batch_size=10, shuffle=False, num_workers=4, drop_last=True)

    print('Source datasets are : ', FLAGS.dataset)

    if 'gta5' in FLAGS.dataset:
        gta5_loader_list = list()
        for dataset in gta5_dataset_list[0:2]:
            gta5_loader_list.append(data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True))
        for dataset in gta5_dataset_list[2:]:
            gta5_loader_list.append(data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4, drop_last=True))

        gta5_iter_list = list()
        for loader in gta5_loader_list:
            gta5_iter_list.append(iter(loader))

        assert(len(gta5_iter_list) == 8)
        assert(len(gta5_loader_list) == 8)
        print('using ', len(gta5_iter_list), ' total variants of GTA5')

    if 'synthia' in FLAGS.dataset:
        synthia_loader_list = list()
        for dataset in synthia_dataset_list[0:2]:
            synthia_loader_list.append(data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True))
        for dataset in synthia_dataset_list[2:]:
            synthia_loader_list.append(data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4, drop_last=True))

        synthia_iter_list = list()
        for loader in synthia_loader_list:
            synthia_iter_list.append(iter(loader))

        assert(len(synthia_iter_list) == 8)
        assert(len(synthia_loader_list) == 8)
        print('using ', len(synthia_iter_list), ' total variants of SYNTHIA')

    if 'synscapes' in FLAGS.dataset:
        synscapes_loader_list = list()
        for dataset in synscapes_dataset_list[0:2]:
            synscapes_loader_list.append(data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True))
        for dataset in synscapes_dataset_list[2:]:
            synscapes_loader_list.append(data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4, drop_last=True))

        synscapes_iter_list = list()
        for loader in synscapes_loader_list:
            synscapes_iter_list.append(iter(loader))

        assert(len(synscapes_iter_list) == 8)
        assert(len(synscapes_loader_list) == 8)
        print('using ', len(synscapes_iter_list), ' total variants of Synscapes')

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    if 'gta5' not in FLAGS.dataset:
        gta5_loader_list = None
        gta5_iter_list = None
    if 'synthia' not in FLAGS.dataset:
        synthia_loader_list = None
        synthia_iter_list = None
    if 'synscapes' not in FLAGS.dataset:
        synscapes_loader_list = None
        synscapes_iter_list = None
    
    list_to_loader = {'gta5': gta5_loader_list, 'synthia': synthia_loader_list, 'synscapes': synscapes_loader_list}
    list_to_iter = {'gta5': gta5_iter_list, 'synthia': synthia_iter_list, 'synscapes': synscapes_iter_list}

    add_img_th = FLAGS.save_every # after _ iterations, add images
    i_lambda = FLAGS.mixup_lambda

    IMG_MEAN = torch.tensor(np.array([[104, 116, 122]], dtype=np.uint8))
    IMG_MEAN = torch.reshape(IMG_MEAN, (1,3,1,1))
    IMG_MEAN = IMG_MEAN.repeat(1,1,512,1024)
    IMG_MEAN = IMG_MEAN.to(device)
    max_mean_IoU = 0.0


    print('started training')

    for iteration in range(FLAGS.start_iter, FLAGS.end_iter):
        gta_model.train()
        if FLAGS.model == 'fcn8s':
            gta_model.adjust_learning_rate(FLAGS, optimizer, iteration)

        for param in gta_model.parameters():
            param.grad = None

        index = iteration

        if index % len(FLAGS.dataset) == 0:
            print('Index number: ', index)
            k = random.randint(0, len(list_to_loader[FLAGS.dataset[0]]) - 1)
            try:
                src_lbl, src_img, src_img_blur = list_to_iter[FLAGS.dataset[0]][k].next()
            except StopIteration:
                gta5_iter_list[k] = iter(list_to_loader[FLAGS.dataset[0]][k])
                src_lbl, src_img, src_img_blur = list_to_iter[FLAGS.dataset[0]][k].next()
        elif index % len(FLAGS.dataset) == 1:
            print('Index number: ', index)
            k = random.randint(0, len(list_to_loader[FLAGS.dataset[1]]) - 1)
            try:
                src_lbl, src_img, src_img_blur = list_to_iter[FLAGS.dataset[1]][k].next()
            except StopIteration:
                synscapes_iter_list[k] = iter(list_to_loader[FLAGS.dataset[1]][k])
                src_lbl, src_img, src_img_blur = list_to_iter[FLAGS.dataset[1]][k].next()
        else:
            print('Index number: ', index)
            k = random.randint(0, len(list_to_loader[FLAGS.dataset[2]]) - 1)
            try:
                src_lbl, src_img, src_img_blur = list_to_iter[FLAGS.dataset[2]][k].next()
            except StopIteration:
                synthia_iter_list[k] = iter(list_to_loader[FLAGS.dataset[2]][k])
                src_lbl, src_img, src_img_blur = list_to_iter[FLAGS.dataset[2]][k].next()

        src_img, src_lbl, src_img_blur = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda(), Variable(src_img_blur).cuda()

        input_size = src_img.size()[2:]


        with torch.no_grad():
            edge_img = edge_model(src_img_blur)
            edge_img = get_edge(edge_img, img_size = (1024, 512), meansub_or_norm = 'meansub')
            edge_img = edge_img.to(device)

        with autocast():

            edge_img = edge_img + 229.84

            src_img += IMG_MEAN

            mixup_img = i_lambda*edge_img + (1-i_lambda)*src_img
            mixup_img = mixup_img

            mixup_img -= IMG_MEAN
            src_img -= IMG_MEAN

            edge_img = edge_img - 229.84

            pred = gta_model(mixup_img)
            pred = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)

            output_sm = F.softmax(pred, dim=1)
            loss = lovasz_softmax(output_sm, src_lbl, ignore=19)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if FLAGS.model == 'deeplab':
            scheduler.step()

        # adding image
        if (index % add_img_th == 0):
            grid = segMap3(src_img, src_lbl, pred)
            writer.add_image('training/rgb_label_pred', grid, index)
            print('iteration : ', index)
            mean_IoU, clean_mIoU = validate_edges(index, cityvalloader, gta_model, edge_model, device, 'meansub', writer, IMG_MEAN, i_lambda=FLAGS.mixup_lambda, dataset_split='cityscapes_val')
            writer.add_scalar("validation/citysc_val_mIoU_mixup", mean_IoU, index)
            writer.add_scalar("validation/citysc_val_mIoU", clean_mIoU, index)

            if(mean_IoU > max_mean_IoU):
                max_mean_IoU = mean_IoU
                print("Saving best (max mean_IoU) model checkpoint")
                torch.save(gta_model.state_dict(), os.path.join('./checkpoints', FLAGS.save_current_model))
            else:
                print('not saving since current miou is ', mean_IoU, 'while best is ', max_mean_IoU)

        if (index % (add_img_th * 2) == 0):
            print('saving for backup')
            torch.save(gta_model.state_dict(), os.path.join('./checkpoints', FLAGS.save_current_model[:-4] + '_backup.pth'))
            print('done saving')

        # adding scalars
        writer.add_scalar('loss/ce_loss', loss.item(), index)
        writer.add_scalar('metrics/train_miou', score['Mean IoU'], index)
