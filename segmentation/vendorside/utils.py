import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms import Compose, ToTensor
from torchvision import transforms
import torch.nn.functional as F
import sys
import numpy as np
import torchvision
from label_to_colormap import create_cityscapes_label_colormap
from sklearn.metrics import confusion_matrix
from PIL import Image
from metrics import StreamSegMetrics


def validate_edges(iteration, loader, model, edge_model, device, meansub_or_norm, tb_writer, IMG_MEAN, i_lambda=0.5, dataset_split='cityscapes_val'):
    
    val_metrics = StreamSegMetrics(n_classes=19)
    val_metrics_rgb = StreamSegMetrics(n_classes=19)
    
    val_metrics.reset()
    val_metrics_rgb.reset()
    
    model.eval() #### eval
    
    with torch.no_grad():  #### eval
        
        for i, (gt_label_batch, rgb_batch, blur_batch) in enumerate(loader):
            
            blur_batch = blur_batch.to(device=device, dtype=torch.float)
            rgb_batch = rgb_batch.to(device=device, dtype=torch.float)
            gt_label_batch = gt_label_batch.to(device=device, dtype=torch.int64)
        
            edge_img = edge_model(blur_batch)
            edge_img = get_edge(edge_img, img_size = (1024, 512), meansub_or_norm = meansub_or_norm)
            edge_img = edge_img.to(device)

            if (meansub_or_norm == 'meansub'):
                edge_img = edge_img + 229.84
            elif (meansub_or_norm == 'norm'):
                edge_img = edge_img * 255   

            rgb_batch += IMG_MEAN

            mixup_img = i_lambda*edge_img + (1-i_lambda)*rgb_batch
            mixup_img = mixup_img

            mixup_img -= IMG_MEAN
            rgb_batch -= IMG_MEAN

            with torch.cuda.amp.autocast():
                output = model(mixup_img)
                pred = model(rgb_batch)                
                    
                output = F.interpolate(output, (512, 1024), mode='bilinear', align_corners=True)                     
                pred = F.interpolate(pred, (512, 1024), mode='bilinear', align_corners=True)
            
            output_np = output.detach().max(dim=1)[1].cpu().numpy()
            pred_np = pred.detach().max(dim=1)[1].cpu().numpy()
            targets = gt_label_batch.cpu().numpy()
            val_metrics.update(targets, output_np)
            val_metrics_rgb.update(targets, pred_np)

            if (meansub_or_norm == 'meansub'):
                edge_img = edge_img - 229.84
            elif (meansub_or_norm == 'norm'):
                edge_img = edge_img / 255   

            if(tb_writer):
                if( i % 150 == 0):
                    grid = segMap_mixup_val(rgb_batch, gt_label_batch, output, edge_img, mixup_img, pred, meansub_or_norm)
                    tb_writer.add_image("validation/"+dataset_split+"/rgb_pred_label_edge_mixup_pred", grid, i + iteration)
                
    score = val_metrics.get_results()
    score_rgb = val_metrics_rgb.get_results()
    
    model.train() #### set to train 
    
    return score['Mean IoU'], score_rgb['Mean IoU']

def segMap_mixup_val(rgb, gt_label, adapted_output, edge_img, mixup_img, rgb_pred, meansub_or_norm):
    
    # plotting for 1st image of batch 
    rgb, gt_label, adapted_output, edge_img, mixup_img, rgb_pred = rgb[0], gt_label[0], adapted_output[0], edge_img[0], mixup_img[0], rgb_pred[0]
    
    
    adapted_output = F.softmax(adapted_output, dim=0)
    adapted_output = torch.argmax(adapted_output, dim=0)
    adapted_output = adapted_output.cpu()
    adapted_output = label2Color(adapted_output)

    rgb_pred = F.softmax(rgb_pred, dim=0)
    rgb_pred = torch.argmax(rgb_pred, dim=0)
    rgb_pred = rgb_pred.cpu()
    rgb_pred = label2Color(rgb_pred)

    
    gt_label = gt_label.cpu()
    gt_label = label2Color(gt_label)
    
#    # denormalize ( edge_img * 255 ) or de - mean subtraction ( edge_img + 229.84 )
    if (meansub_or_norm == 'meansub'):
        edge_img = edge_img + 229.84
        
    elif (meansub_or_norm == 'norm'):   
        edge_img = edge_img * 255 
    
    # convert grayscale edge_img to 3 channels
    edge_img = edge_img.repeat(3, 1, 1) 
    edge_img = edge_img.permute(1,2,0)
    edge_img = edge_img.cpu()
    edge_img = np.asarray(edge_img, dtype=np.uint8)

    
    mixup_img = mixup_img.permute(1,2,0)
    mixup_img = mixup_img.cpu()
    mixup_img = np.asarray(mixup_img, dtype=np.uint8)
    IMG_MEAN = np.array((104, 116, 122), dtype=np.uint8)
    mixup_img += IMG_MEAN
    mixup_img = mixup_img[:, :, : : -1]

    rgb = rgb.permute(1,2,0)
    rgb = rgb.cpu()
    rgb = np.asarray(rgb, dtype=np.uint8)
    IMG_MEAN = np.array((104, 116, 122), dtype=np.uint8)
    rgb += IMG_MEAN
    rgb = rgb[:, :, : : -1]

    grid = torch.from_numpy(np.asarray([rgb, rgb_pred, gt_label, edge_img, mixup_img, adapted_output]))
    grid = grid.permute(0,3,1,2)
    
    grid = torchvision.utils.make_grid(grid)


    return grid

def get_edge(tensor, img_size=(1024, 512), meansub_or_norm='meansub'):
    
    batch_size = tensor[0].size()[0]

    edge_maps = []
    
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
         
    tensor = np.array(edge_maps)
    
    image_shape = [img_size] * batch_size
    
    edge_imgs = torch.zeros([batch_size, 1, img_size[1], img_size[0]])
    
    idx = 0
    
    for i_shape in image_shape:
        
        tmp = tensor[:, idx, ...]
        
        # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
        tmp = np.squeeze(tmp)


        # Iterate our all 7 NN outputs for a particular image
        preds = []
        
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            tmp_img[tmp_img < 0.0] = 0.0
            tmp_img = 255.0 * (1.0 - tmp_img)

            # Resize prediction to match input image size
            if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))

            preds.append(tmp_img)
            if i == 6:
                fuse = tmp_img

        fuse = fuse.astype(np.uint8)

        # Get the mean prediction of all the 7 outputs
        average = np.array(preds, dtype=np.float32)
        average = np.uint8(np.mean(average, axis=0))
        
        if(meansub_or_norm == 'meansub'):
            average = average - 229.84
        
        elif(meansub_or_norm == 'norm'):
            average = average / 255
        
        
        edge_imgs[idx] = torch.tensor(average)

        idx += 1


    return edge_imgs

def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image

def weightsInit(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()

def label2Color(label):
    label = np.asarray(label, dtype=np.uint8)
    colormap = create_cityscapes_label_colormap()
    image = np.zeros((label.shape[0],label.shape[1],3), dtype=np.uint8)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if(label[i,j] > 19):
                label[i,j] = 19
            image[i,j] = colormap[label[i,j]]
    return image

def visualizeOneHot(gt, predict, embed=None):
    gt = gt.cpu()
    predict = predict.cpu()

    gt = gt.permute(0,2,3,1).numpy()
    predict = predict.permute(0,2,3,1).detach().numpy()

    gt_imgs = np.argmax(gt, axis=3)
    predict_imgs = np.argmax(predict, axis=3)
    temp = torch.from_numpy(np.asarray([label2Color(gt_imgs[0]), label2Color(predict_imgs[0])]))

    if(embed != None):
        embed = embed.cpu()
        embed = embed.permute(0,2,3,1).detach().numpy()
        embed_imgs = np.argmax(embed, axis=3)
        temp = torch.from_numpy(np.asarray([label2Color(gt_imgs[0]), label2Color(predict_imgs[0]), label2Color(embed_imgs[0])]))



    temp = temp.permute(0,3,1,2)
    img_grid = torchvision.utils.make_grid(temp)

    return img_grid


def postProcessLabel(label_predict, label_gt):
    h,w = label_gt.shape
    for y in range(h):
        for x in range(w):
            if(label_gt[y,x] == 19):
                label_predict[y,x] = 19
    return label_predict


def segMaps0(img_label, img_fake):
    # plotting for 0th batch only
    img_label, img_fake = img_label[0], img_fake[0]
    img_fake = F.softmax(img_fake, dim=0)
    img_fake = torch.argmax(img_fake, dim=0)

    img_label = img_label.cpu()
    img_fake = img_fake.cpu()

    img_label = label2Color(img_label)
    img_fake = label2Color(img_fake)

    grid = torch.from_numpy(np.asarray([img_label, img_fake]))
    grid = grid.permute(0,3,1,2)
    grid = torchvision.utils.make_grid(grid)

    return grid


def segMaps(img_label, img_fake, img_real):
    # plotting for 0th batch only
    img_label, img_fake, img_real = img_label[0], img_fake[0], img_real[0]
    img_fake = F.softmax(img_fake, dim=0)
    img_fake = torch.argmax(img_fake, dim=0)
    img_real = F.softmax(img_real, dim=0)
    img_real = torch.argmax(img_real, dim=0).squeeze(0)

    img_label = img_label.cpu()
    img_fake = img_fake.cpu()
    img_real = img_real.cpu()

    img_label = label2Color(img_label)
    img_fake = label2Color(img_fake)
    img_real = label2Color(img_real)

    grid = torch.from_numpy(np.asarray([img_label, img_fake, img_real]))
    grid = grid.permute(0,3,1,2)
    grid = torchvision.utils.make_grid(grid)

    return grid

def segMaps2(img_label, img_fake):
    # plotting for 0th batch only
    img_label, img_fake = img_label[0], img_fake[0]
    img_fake = F.softmax(img_fake, dim=0)
    img_fake = torch.argmax(img_fake, dim=0)

    img_label = img_label.cpu()
    img_fake = img_fake.cpu()

    img_label = label2Color(img_label)
    img_fake = label2Color(img_fake)

    grid = torch.from_numpy(np.asarray([img_label, img_fake]))
    grid = grid.permute(0,3,1,2)
    grid = torchvision.utils.make_grid(grid)

    return grid

def segMap3(rgb, img_label, pred):
    # plotting for 0th batch only
    rgb, img_label, pred = rgb[0], img_label[0], pred[0]
    rgb = rgb.permute(1,2,0)

    pred = F.softmax(pred, dim=0)
    pred = torch.argmax(pred, dim=0)

    img_label = img_label.cpu()
    rgb = rgb.cpu()
    pred = pred.cpu()

    img_label = label2Color(img_label)
    pred = label2Color(pred)
    rgb = np.asarray(rgb, dtype=np.uint8)
    IMG_MEAN = np.array((104, 116, 122), dtype=np.uint8)
    rgb += IMG_MEAN
    rgb = rgb[:, :, : : -1]

    grid = torch.from_numpy(np.asarray([rgb, img_label, pred]))
    grid = grid.permute(0,3,1,2)
    grid = torchvision.utils.make_grid(grid)

    return grid


def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src


def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)     # get b
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate( amp_src.clone(), amp_trg.clone(), L=L )

    # recompose fft of source
    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg

##### AdaIN helper functions
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


# only for single style image and single content image
def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)
