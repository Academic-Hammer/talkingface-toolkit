实验使用的数据集是CSTR VCTK Corpus数据集，可以在下面的地址中下载获得。
- [CSTR VCTK Corpus](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)

在dataset文件夹中，VCTK-Corpus存放的是在网站下载的原始数据，preprocessed_data存放的是经过预处理之后的数据。

经过预处理后的dataset的架构如下：
```xml
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