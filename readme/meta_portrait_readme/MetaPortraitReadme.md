# 基本介绍
本模型整合模型的是"MetaPortrait: Identity-Preserving Talking Head Generation with Fast Personalized Adaptation(CVPR 2023)"论文提出的MetaPortrait模型，主要代码整合自[该仓库](https://github.com/Meta-Portrait/MetaPortrait#temporal-super-resolution-model)

# 环境设置
为了保证和原仓库代码的兼容性，模型代码基本按照原仓库环境配置
```bash
cd talkingface
conda env create -f environment.yml
conda activate cszhouvoice
```

# 基础模型使用

## Inference Base Model
在运行推理代码之前需要首先下载[模型checkpoint](https://drive.google.com/file/d/1Kmdv3w6N_we7W7lIt6LBzqRHwwy1dBxD/view?usp=share_link)并将其放在`saved`文件夹下，下载[示例数据](https://drive.google.com/file/d/166eNbabM6TeJVy7hxol2gL1kUGKHi3Do/view?usp=share_link)解压放在`dataset/meta_portrait_base_data`下。

处理好后文件结构如下：
```
talkingface
├───saved
│   └───ckpt_base.pth.tar
└───dataset
    └───meta_portrait_base_data
        ├── 0
        │   ├── imgs
        │   │   ├── 00000000.png
        │   │   ├── ...
        │   ├── ldmks
        │   │   ├── 00000000_ldmk.npy
        │   │   ├── ...
        │   └── thetas
        │       ├── 00000000_theta.npy
        │       ├── ...
        ├── src_0_id.npy
        ├── src_0_ldmk.npy
        ├── src_0.png
        ├── src_0_theta.npy
        └── src_map_dict.pkl
```

可以通过以下命令生成256*256分辨率的self reconstruction的结果
```bash
python run_talkingface.py --model==metaportraitbaseinference --save_dir ./saved --config ./talkingface/properties/model/meta_portrait_base_config/meta_portrait_256_eval.yaml --ckpt ./saved/ckpt_base.pth.tar
```
推理成功后结果将保存在saved文件夹下，将生成output文件，推理成功截图见`readme/meta_portrait_readme/inference.png`

## Train Base Model from Scratch

首先需要使用以下命令训练warping网络
```bash
python run_talkingface.py --model==metaportraitbasetrain --config ./talkingface/properties/model/meta_portrait_base_config/meta_portrait_256_pretrain_warp.yaml --fp16 --stage Warp --task Pretrain
```
训练成功后将生成.pth.tar的checkpoint，其结果保存在`saved/ckpt_warp/meta_portrait_base`下，训练成果截图见`readme/meta_portrait_readme/warp_train1.png`和`readme/meta_portrait_readme/warp_train2.png`

在继续训练前需要从上一步训练结果`saved/ckpt_warp/meta_portrait_base`选择合适的checkpoint，将其重命名为`ckpt_warp.pth.tar`并将该文件移动到`saved`文件夹下（或者也可以直接修改config文件下的路径参数）。然后运行以下命令对整个网络进行训练。
```bash
python run_talkingface.py --model==metaportraitbasetrain --config ./talkingface/properties/model/meta_portrait_base_config/meta_portrait_256_pretrain_full.yaml --fp16 --stage Full --task Pretrain
```
训练成功后将生成.pth.tar的checkpoint，其结果保存在`saved/ckpt_full/meta_portrait_base`下，训练成果截图见`readme/meta_portrait_readme/full_train1.png`和`readme/meta_portrait_readme/full_train2.png`

## Meta Training for Faster Personalization of Base Model
在继续训练前需要从上一步训练结果`saved/ckpt_full/meta_portrait_base`选择合适的checkpoint，将其重命名为`ckpt_full.pth.tar`并将该文件移动到`saved`文件夹下（或者也可以直接修改config文件下的路径参数）。然后运行以下命令对整个网络进行元训练进而对网络进行个性化调优。
```bash
python run_talkingface.py --model==metaportraitbasetrain --config ./talkingface/properties/model/meta_portrait_base_config/meta_portrait_256_meta_train.yaml --fp16 --stage Full --task Meta --remove_sn --ckpt ./saved/ckpt_full.pth.tar
```
训练成功后将生成.pth.tar的checkpoint，其结果保存在`saved/ckpt_meta_train/meta_portrait_meta_train`下，训练成果截图见`readme/meta_portrait_readme/meta_train1.png`和`readme/meta_portrait_readme/meta_train2.png`