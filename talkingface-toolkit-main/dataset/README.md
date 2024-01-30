这个文件夹中保存的是数据集如：
- lrw
- lrs2
- mead
- .......

数据集处理的一般格式为：

```
dataset/

├── lrs2

| ├── data (存放数据集的原始数据)

| ├── filelist (保存的是数据集划分)

| │ ├── train.txt

| │ ├── val.txt

| │ ├── test.txt

| ├── preprocessed_data (具体路径内容可以参考talkingface.utils.data_preprocess文件中处理lrs2时候的路径，主要存储的是视频抽帧后的图像文件和音频文件)
```

preprocessed_data的数据路径一般表示为：
```
preprocessed_root (lrs2_preprocessed)/

├── list of folders

| ├── Folders with five-digit numbered video IDs

| │ ├── *.jpg

| │ ├── audio.wav

```

数据集存储尽量按照这个格式来，数据集的划分也尽量按照train.txt val.txt和test.txt文件来
