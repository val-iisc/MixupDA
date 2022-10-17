from dataset import CityscapesDataset, CityscapesDataset_blur, GTA5Dataset, GTA5Dataset_blur, SynthiaDataset, SynthiaDataset_blur

def get_synscapes_datasets_blur(list_path, mirror=True):
    transform = False
    adain_gta_dataset = GTA5Dataset_blur(
        root='./datasets/synscapes/img/class',
        rgb_root='./datasets/synscapes/img/rgb',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        adain=0.3,
        mirror=mirror,
        dataset='synscapes'
    )

    cvprw_gta_dataset = GTA5Dataset_blur(
        root='./datasets/synscapes/img/class',
        rgb_root='./datasets/synscapes/img/rgb',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        styleaug=True,
        mirror=mirror,
        dataset='synscapes'
    )

    fda_style_gta_dataset = GTA5Dataset_blur(
        root='./datasets/synscapes/img/class',
        rgb_root='./datasets/synscapes/img/rgb',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        fda=('input/style', 0.005),
        mirror=mirror,
        dataset='synscapes'
    )

    fda_random_gta_dataset = GTA5Dataset_blur(
        root='./datasets/synscapes/img/class',
        rgb_root='./datasets/synscapes/img/rgb',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        fda=('random', 0.005),
        mirror=mirror,
        dataset='synscapes'
    )

    snow_gta_dataset = GTA5Dataset_blur(
        root='./datasets/synscapes/img/class',
        rgb_root='./datasets/synscapes/img/rgb',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        imgaug='snow',
        mirror=mirror,
        dataset='synscapes'
    )

    frost_gta_dataset = GTA5Dataset_blur(
        root='./datasets/synscapes/img/class',
        rgb_root='./datasets/synscapes/img/rgb',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        imgaug='frost',
        mirror=mirror,
        dataset='synscapes'
    )

    cartoon_gta_dataset = GTA5Dataset_blur(
        root='./datasets/synscapes/img/class',
        rgb_root='./datasets/synscapes/img/rgb',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        imgaug='cartoon',
        mirror=mirror,
        dataset='synscapes'
    )

    datasets = [
        adain_gta_dataset,
        cvprw_gta_dataset,
        fda_style_gta_dataset,
        fda_random_gta_dataset,
        snow_gta_dataset,
        frost_gta_dataset,
        cartoon_gta_dataset,
    ]

    return datasets


def get_gta5_datasets_blur(list_path):
    mirror = True
    transform = False
    adain_gta_dataset = GTA5Dataset_blur(
        root='./datasets/gta5-dataset',
        rgb_root='./datasets/gta5-dataset',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        adain=0.3,
        mirror=mirror
    )
    
    cvprw_gta_dataset = GTA5Dataset_blur(
        root='./datasets/gta5-dataset',
        rgb_root='./datasets/gta5-dataset',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        styleaug=True,
        mirror=mirror
    )
    
    fda_style_gta_dataset = GTA5Dataset_blur(
        root='./datasets/gta5-dataset',
        rgb_root='./datasets/gta5-dataset',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        fda=('input/style', 0.005),
        mirror=mirror
    )
    
    fda_random_gta_dataset = GTA5Dataset_blur(
        root='./datasets/gta5-dataset',
        rgb_root='./datasets/gta5-dataset',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        fda=('random', 0.005),
        mirror=mirror
    )
    
    snow_gta_dataset = GTA5Dataset_blur(
        root='./datasets/gta5-dataset',
        rgb_root='./datasets/gta5-dataset',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        imgaug='snow',
        mirror=mirror
    )
    
    frost_gta_dataset = GTA5Dataset_blur(
        root='./datasets/gta5-dataset',
        rgb_root='./datasets/gta5-dataset',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        imgaug='frost',
        mirror=mirror
    )
    
    cartoon_gta_dataset = GTA5Dataset_blur(
        root='./datasets/gta5-dataset',
        rgb_root='./datasets/gta5-dataset',
        list_path=list_path,
        crop_size=(1024, 512),
        transform=transform,
        imgaug='cartoon',
        mirror=mirror
    )

    datasets = [
        adain_gta_dataset,
        cvprw_gta_dataset,
        fda_style_gta_dataset,
        fda_random_gta_dataset,
        snow_gta_dataset,
        frost_gta_dataset,
        cartoon_gta_dataset,
    ]

    return datasets

def get_synthia_datasets_blur(list_path, crop='random', mirror=True):
    adain_synth_dataset = SynthiaDataset_blur(
        root='./datasets/synthia_cityscape/RAND_CITYSCAPES',
        list_path=list_path,
        mirror=mirror,
        crop=crop,
        adain=0.3
    )

    cvprw_synth_dataset = SynthiaDataset_blur(
        root='./datasets/synthia_cityscape/RAND_CITYSCAPES',
        list_path=list_path,
        mirror=mirror,
        crop=crop,
        styleaug=True
    )

    fda_style_synth_dataset = SynthiaDataset_blur(
        root='./datasets/synthia_cityscape/RAND_CITYSCAPES',
        list_path=list_path,
        mirror=mirror,
        crop=crop,
        fda=('input/style', 0.005)
    )

    fda_rand_synth_dataset = SynthiaDataset_blur(
        root='./datasets/synthia_cityscape/RAND_CITYSCAPES',
        list_path=list_path,
        mirror=mirror,
        crop=crop,
        fda=('random', 0.005)
    )

    snow_synth_dataset = SynthiaDataset_blur(
        root='./datasets/synthia_cityscape/RAND_CITYSCAPES',
        list_path=list_path,
        mirror=mirror,
        crop=crop,
        imgaug='snow'
    )

    frost_synth_dataset = SynthiaDataset_blur(
        root='./datasets/synthia_cityscape/RAND_CITYSCAPES',
        list_path=list_path,
        mirror=mirror,
        crop=crop,
        imgaug='frost'
    )

    cartoon_synth_dataset = SynthiaDataset_blur(
        root='./datasets/synthia_cityscape/RAND_CITYSCAPES',
        list_path=list_path,
        mirror=mirror,
        crop=crop,
        imgaug='cartoon'
    )

    datasets = [
        adain_synth_dataset,
        cvprw_synth_dataset,
        fda_style_synth_dataset,
        fda_rand_synth_dataset,
        snow_synth_dataset,
        frost_synth_dataset,
        cartoon_synth_dataset,
    ]

    return datasets
