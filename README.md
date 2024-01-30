# talkingface-toolkit小组作业
小组成员名单：
陈清扬  高艺芙  贺芳琪  

## 模型选择
我们选择复现的模型是[StarGAN-VC](https://github.com/kamepong/StarGAN-VC)模型，是专为语音转换任务设计的模型。它是StarGAN框架的延伸，该框架最初用于图像到图像的转换任务。StarGAN-VC专注于将一个说话者的语音特征转换为另一个说话者。



## 作业环境
 



## 数据集
 


## 运行说明
 
### 运行指令


python run_talkingface.py --model stargan --dataset vctk



## 实现功能
train_stargan_model.py
###Imports: 
Import necessary libraries and modules, including PyTorch, TorchVision, and other utilities.

###Data Preprocessing:

walk_files Function: Recursively walks through a directory and yields file paths with a specified extension.
logmelfilterbank Function: Computes mel spectrogram features from audio data using librosa.
extract_melspec Function: Extracts mel spectrogram features from audio files and saves them in HDF5 format.

###Feature Normalization:

normalize_features Function: Normalizes mel spectrogram features using StandardScaler and saves the normalized features in HDF5 format.

###Compute Statistics:

compute_statistics Function: Computes and saves statistics (mean and standard deviation) of mel spectrogram features for normalization.

###Model Training:

train_stargan Function: Defines the StarGAN model architecture (generator and discriminator) and trains the model.
The script uses a parallelized approach (joblib) to extract mel spectrogram features and normalize them.
The training loop includes the generator and discriminator updates, loss computation, and logging of training progress.

###Configuration and Logging:

Various parameters such as data paths, model configurations, and training hyperparameters are set at the beginning of the script.
Logging is used to record training progress, and a configuration file is saved for reproducibility.

###Main Execution:

The script concludes with the main execution of the train_stargan function.



run_talkingface.py
###Import necessary modules:

argparse: Used for parsing command-line arguments.
train_stargan_model from models.stargan: Presumably, this function is responsible for training the StarGAN model. (Note: The import is commented out.)

###Define the run function:

Takes in arguments like model_name, dataset_name, config_file_list, evaluate_model_file, and train.
Checks if the train flag is True and if the model_name is 'stargan', then calls the train_stargan_model function.

###Check if the script is being run as the main module:

If true, use argparse to parse command-line arguments.
Set default values for various arguments.
Split the config_files argument into a list if provided.
Call the run function with the parsed arguments.







## 结果截图


![](./md_img/1.jpg)

![](./md_img/2.jpg)

![](./md_img/3.jpg)

![](./md_img/4.jpg)



## 所用依赖





## 成员分工
陈清扬-1120213599-07112106：模型代码重构，训练代码重构，推理文件重构，配置文件调整，测试代码与bug修复，撰写实验报告。

高艺芙-1120213132-07022102：负责配置好实验环境，准备实验数据，运行StarGan模型，将得到配置文件json转换为yaml并整合到相应的文件结构中。

贺芳琪-1120210640-08012101：部分配置调整，主要负责data，数据预处理部分




## 花絮

数据集太大了实在不好上传到github就删掉了
