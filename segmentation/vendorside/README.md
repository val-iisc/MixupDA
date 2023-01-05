## Vendor-side (source-only) training

1. Download model weights from [Google Drive](https://drive.google.com/drive/folders/1IgDmX4jtKV9SP3SEqpVhxP4Rhbx33q7c?usp=sharing) and copy it to `checkpoints`. 
2. Set training arguments in ``script.sh`` file. Some important arguments are mentioned below:
    - `CUDA_VISIBLE_DEVICES`: GPU ID to be used for training.
    - `--dataset`: specify `gta5` or `synthia` or `gta5 synthia` (multi-source case with GTA5+SYNTHIA) or any combination of `gta5`, `synthia` and `synscapes` for vendor-side (single-source or multi-source) training.
    - `--load_prev_model`: specify the name of checkpoint to be loaded as initialization (should be stored in `./checkpoints` folder).
    - `--save_current_model`: specify the name for checkpoint that will be saved during training.
    - `--save_every`: model is saved after every `save_every` number of iterations.
    - `--runs`: specify name of experiment so the corresponding log files saved in `./runs/` folder can be properly visualized in Tensorboard.
    - `--mixup_lambda`: specify the value of $\lambda$ in the edge-RGB mixup (0.005 by default). Can be set to 0 for ICCV21 baselines.
3. Start training by running ``bash script_edge_mixup.sh`` command within the `vendorside` folder.
4. To evaluate the model, run ``bash eval.sh`` after setting the name of the checkpoint to be evaluated in `eval.sh`. Refer to the README file in the `segmentation` folder for more details.
