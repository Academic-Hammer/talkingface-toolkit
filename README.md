# Adaptive Voice Conversion

## Abstract

- 实验项目为voice_conversion类别下的adaptive-VC；

- 项目由付宇轩1120213142个人完成
  - 重构代码地址：https://github.com/FFFXX0319/talkingface-toolkit.git
  - 源代码地址：https://github.com/jjery2243542/adaptive_voice_conversion.git

- 文档主要分为两个部分：
  - 代码结构介绍：对重构的代码结构进行介绍，并说明完成的工作
  - 实验步骤：说明实验流程，对模型训练和验证的结果进行展示

## 代码结构介绍

### checkpoints                                                                                                                                               

- the pretrain model   

  http://speech.ee.ntu.edu.tw/~jjery2243542/resource/model/is19/vctk_model.ckpt

- the coresponding normalization parameters for inference 

  http://speech.ee.ntu.edu.tw/~jjery2243542/resource/model/is19/attr.pkl

### dataset

实验使用的数据集是CSTR VCTK Corpus数据集，可以在下面的地址中下载获得。

\- [CSTR VCTK Corpus](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)

在dataset文件夹中，VCTK-Corpus存放的是在网站下载的原始数据，preprocessed_data存放的是经过预处理之后的数据。

```python
dataset
├── preprocessed_data *
│   ├── attr.pkl
│   ├── in_test_files.txt
│   ├── in_test.pkl
│   ├── in_test_samples_128.json
│   ├── out_test_files.txt
│   ├── out_test.pkl
│   ├── out_test_samples_128.json
│   ├── train_128.pkl
│   ├── train.pkl
│   └── train_samples_128.json
├── VCTK-Corpus *
│   ├── COPYING
│   ├── NOTE
│   ├── README
│   ├── speaker-info.txt
│   ├── txt *
│   └── wav48 *
└── (*符号代表该路径名为文件夹)
```

### result

result文件夹存放了模型训练和验证的log文件，以及模型生成的语音wav文件。

```python
result
├── evaluate
│   ├── source.wav   
│   ├── target.wav
│   ├── eval.wav
│   └── evaluate_output.log
└── train
    └── train_output.log
```

### saved

训练过程中保存的模型checkpoint, 训练过程中保存模型时自动创建。

```python
saved
└── adaptive_vc
    ├── vctk_model.args.yaml   # 训练时的命令参数配置文件              
    ├── vctk_model.ckpt        # 训练后模型的checkpoint
    ├── vctk_model.config.yaml   # 模型的参数配置文件
    └── vctk_model.opt
```

### tf-logs

训练的日志文件，用于tensorboard绘制评估曲线。

### talkingface模块

模型的核心代码部分

#### config

根据模型和数据集名称自动生成所有模型、数据集、训练、评估等相关的配置信息。

#### data

```python
data
├── dataprocess   # 模型特有的数据处理代码
│   ├── __init__.py
│   ├── make_datasets_vctk.py
│   ├── preprocess_vctk.sh
│   ├── reduce_dataset.py
│   ├── sample_segments.py
│   ├── sample_single_segments.py
│   ├── tacotron
│   │   ├── hyperparams.py
│   │   └── utils.py
│   └── vctk.config
└── dataset
    ├── __init__.py
    ├── dataset.py
    └── vctk_dataset.py

```

- dataprocess：模型特有的数据处理代码，可以通过运行preprocess_vctk.sh脚本直接进行数据的处理。

- dataset：
  - vctk_dataset.py:  创建SequenceDataset，PickleDataset， get_data_loader等子类，继承自基类Dataset，实现基类中的方法。

#### model



#### properties



#### utils



#### trainer



#### evalutor



## 实验步骤

### 1. 环境搭建

- 硬件环境
  - CPU ：12 核心   内存：43 GB
  - GPU ：NVIDIA GeForce RTX 2080 Ti

实验在Autodl平台上进行模型的训练和评估，选择使用平台提供的miniconda3镜像，按照以下步骤搭建环境。

- **创建conda虚拟环境py36并激活环境**

  ```shell
  conda create -n py36 python=3.6
  conda init bash && source /root/.bashrc
  conda activate py36
  ```

- **查看显卡支持的最大cuda版本，选择合适的pytorch版本安装**

  ```assembly
  nvidia-smi
  ```

  ![image-20240119193402509](README.assets/image-20240119193402509.png)

- **安装实验所用的第三方库**

  ​		实验根据文档提出的环境要求进行了尝试，使用`python=3.8`，`torch==1.13.1+cu116`，`numpy==1.20.3`，`librosa==0.10.1`进行环境配置，但是在项目的复现中遇到了非常多的版本冲突。项目源码实现的环境使用了低于0.8.0版本的librosa，低于0.50.0版本的numba，使用文档要求的环境会引发一系列与numba\llvmlite\python包版本不兼容的问题，所以实验使用了较早版本的第三方库环境，具体的第三方库配置已导出到requirement.txt中。

  - torch=1.0.1  torchvision=0.2.2
  - numba=0.48.0
  - numpy=1.19.5
  - pip=21.3.1
  - SoundFile=0.10.2
  - librosa=0.7.2
  - llvmlite=0.31.0

### 2.数据预处理

对数据集进行预处理，将数据集划分为train_set、 in_test_set和out_test_set。

```bash
bash preprocess_vctk.sh
```

![image-20240118205212486](README.assets/image-20240118205212486.png)

经过预处理后的数据架构如下：

![image-20240119123956865](README.assets/image-20240119123956865.png)

### 3.模型训练

使用run_talkingface.py进行模型的训练

```
python run_talkingface.py --model=Adaptive_VC --dataset=VCTK
```

训练参数设置如下：

- **--model**, default="Adaptive_VC"
- **--dataset**, default="VCTK"

- **--config**,  default='./talkingface/properties/model/adaptive-VC.yaml'

- **--data_dir**, default='./autodl-fs/preprocessed_data'

- **--train_set**, default='train_128'

- **--train_index_file**,  default='train_samples_128.json'

- **--logdir**,  default='./tf-logs/'

- **--store_model_path**,  default='./saved/adaptive_vc/vctk_model'

- **--load_model_path**,  default='./saved/adaptive_vc/vctk_model'

- **--summary_steps**,  default=500

- **--save_steps**,  default=5000

- **--iters**,  default=20000

实验的训练结果输出在`result\train\train_output.log`，训练得到的模型在`saved\adaptive_vc\vctk_model.ckpt`。

模型训练过程的截图如下:

![image-20240119181755678](README.assets/image-20240119181755678.png)

经过20000个iter的训练之后，得到的模型效果如下：

![image-20240121232546501](README.assets/image-20240121232546501.png)

![image-20240121231348020](README.assets/image-20240121231348020.png)

### 4.模型验证

使用run_evaluator.py进行模型的验证

```
python run_evaluator.py
```

模型验证的参设置如下：

- **-c**, the path of config file.
- **-m**:,the path of model checkpoint.
- **-a**, the attribute file for normalization ad denormalization.
- **-s**, the path of source file (.wav).
- **-t**, the path of target file (.wav).
- **-o**, the path of output converted file (.wav).

验证使用的模型为`saved/adaptive_vc/vctk_model.ckpt`，模型验证结果的输出在`result\evaluate\evaluate_output.log`，生成的语音在`result\evaluate\eval.wav`。

## 作业要求

- 确保可以仅在命令行输入模型和数据集名称就可以训练、验证。（部分仓库没有提供训练代码的，可以不训练）
- 每个组都要提交一个README文件，写明==完成的功能==、最终**实现的训练、验证截图**、**所使用的依赖**、**成员分工**等。



