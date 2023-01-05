# Experiments on DA for Semantic Segmentation

## Requirements
* Python 3.6+ (recommended, we have not tested the code with previous versions)
* [PyTorch](https://pytorch.org/) 1.6+ (for mixed precision training)
* `imgaug` and `imagecorruptions` libraries (refer their [installation instructions](https://imgaug.readthedocs.io/en/latest/source/installation.html))
* For simplicity, we provide a `environment.yml` file extracted from our conda environment. Install using
```
conda env create -f environment.yml
conda activate daseg
```

## Preparation
### Downloading datasets
- Download the [Cityscapes dataset](https://www.cityscapes-dataset.com/)
- Download the [GTA5 dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)
- Download the [SYNTHIA dataset](http://synthia-dataset.net/download/808/) (RAND-CITYSCAPES version)
- Download the [SYNTHIA processed labels](https://github.com/valeoai/DADA/releases/tag/v0.1)
- Download the [Synscapes dataset](https://synscapes.on.liu.se/download.html)

### Preparing datasets for training/evaluation
- Create dataset symlinks for GTA5, SYNTHIA, Synscapes, and Cityscapes inside `datasets` folder:
```
ln -s /path/to/cityscape ./datasets/cityscape
ln -s /path/to/gta5 ./datasets/gta5-dataset
ln -s /path/to/synthia ./datasets/synthia_cityscape
ln -s /path/to/synscapes ./datasets/synscapes
```

### Pretrained weights
Pretrained weights can be downloaded and copied to the `checkpoints` folder in either `vendorside` or `clientside` (coming soon) folder as required.
- Weights from this paper's results - [Google Drive](https://drive.google.com/drive/folders/1Q3fvDPagFIPT0jXBRX-5hW5pNTW_uChq?usp=sharing)
- Weights from ICCV21 baseline - [Google Drive](https://drive.google.com/drive/folders/1MYZq6DPK6xemSM1yBz3UkwRvBW9rot5q?usp=share_link) 

## Evaluation
- Use `bash eval.sh` within `vendorside` folder to evaluate any saved model weights.
- Set arguments appropriately in `eval.sh` file. The important arguments are:
    - `CUDA_VISIBLE_DEVICES`: GPU ID to be used for training.
    - `model`: specify `deeplab` or `fcn` for model architecture.
    - `dataset`: specify `cityscapes` for evaluating on target data.
    - `load_model`: specify path to model weights to be evaluated, e.g. `'./checkpoints/dl_allg_gta5.pth'`

## Training
- Refer to the specific README files in `vendorside` and `clientside` folders.

### Acknowledgements
We are thankful to [FDA](https://github.com/YanchaoYang/FDA), [DADA](https://github.com/valeoai/DADA), [BDL](https://github.com/liyunsheng13/BDL) and [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet) for releasing their code.

## Citation
If you find our work helpful in your research, please cite the following paper:
```
@InProceedings{pmlr-v162-kundu22a,
  title = 	 {Balancing Discriminability and Transferability for Source-Free Domain Adaptation},
  author =       {Kundu, Jogendra Nath and Kulkarni, Akshay R and Bhambri, Suvaansh and Mehta, Deepesh and Kulkarni, Shreyas Anand and Jampani, Varun and Radhakrishnan, Venkatesh Babu},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {11710--11728},
  year = 	 {2022},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
}
```
