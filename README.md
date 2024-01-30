
# StyleHeat TalkingFace模型实现说明文档
## 目录
- [StyleHeat TalkingFace模型实现说明文档](#styleheat-talkingface模型实现说明文档)
  - [目录](#目录)
  - [简介](#简介)
  - [团队成员分工](#团队成员分工)
  - [快速使用](#快速使用)
    - [预训练模型下载](#预训练模型下载)
    - [模型推理](#模型推理)
      - [视频重演](#视频重演)
      - [图片重演](#图片重演)
      - [音频重演](#音频重演)
    - [模型训练](#模型训练)
      - [VOX-StyleHeatWarpper](#vox-styleheatwarpper)
      - [HDTF-StyleHeat](#hdtf-styleheat)
  - [模型实现](#模型实现)
## 简介

本文档是（王菁芃,周楚舒,明楷,杨梓,潘静雯）小组成果中StyleHeat模型部分的说明文档。metaportrait模型部分在[metaportrait]()仓库中。

StyleHeat是一种基于预训练 StyleGAN 模型的新型统一模型，它利用 StyleGAN 模型中出色的空间变换属性等潜在特征空间，实现一系列强大的功能，即高分辨率视频生成、通过驱动视频或音频进行自由控制以及灵活的面部视频生成。

本项目是StyleHeat模型的在talkingface框架中的整合实现，部分代码取自仓库[StyleHEAT](https://github.com/OpenTalker/StyleHEAT)。

## 团队成员分工

本小组完成了metaportrait和styleheat模型修改工作，具体分工如下：

- **周楚舒**：统筹分工与时间安排，阅读metaportrait论文，配置实验环境，参与小组讨论，跑通了metaportrait代码，并将metaportrait代码的base模块接入toolkit，编写metaportrait模型的说明文档
- **杨   梓**：阅读metaportrait论文，配置实验环境，参与小组讨论，跑通了metaportrait源代码，完成数据集整理和重构，对temporal super-resolution model模块代码重构，部分接口撰写和调试
- **潘静雯**：阅读metaportrait论文和源代码，配置实验环境，参加小组讨论和sr_model的调试工作，完成Temporal Super-resolution Model模型的数据集下载和预处理，撰写实验报告
- **王菁芃**：阅读styleheat论文，配置实验环境，参与小组讨论，进行数据集下载和预处理，负责模型推理与训练部分接口的调试与修改，编写styleheat模型说明文档
- **明   楷**：阅读styleheat论文，配置实验环境，参与小组讨论，代码修改调试，撰写实验报告，进行loss计算

## 快速使用

使用python运行run_talkingface.py脚本即可。在参数中需给出模型名称：StyleHeat，数据集styleheat_vox。例如：

```bash
python run_talkingface.py --model=StyleHeat --dataset=styleheat_vox
```

若不再给出额外参数，则默认进行模型训练。

### 预训练模型下载

- 进行模型推演请下载[预训练模型](https://drive.google.com/drive/folders/1-m47oPsa3kxjgK5eSJ8g8sHzG4zr2oRc)，并放入checkpoints子目录下。

  <details>
      <figure class='table-figure'><table>
  <thead>
  <tr><th>checkpoints/Encoder_e4e.pth</th><th>Pre-trained E4E StyleGAN Inversion Encoder.</th></tr></thead>
  <tbody><tr><td>checkpoints/hfgi.pth</td><td>Pre-trained HFGI StyleGAN Inversion Encoder.</td></tr><tr><td>checkpoints/StyleGAN_e4e.pth</td><td>Pre-trained StyleGAN.</td></tr><tr><td>checkpoints/ffhq_pca.pt</td><td>StyleGAN editing directions.</td></tr><tr><td>checkpoints/ffhq_PCA.npz</td><td>StyleGAN optimization parameters.</td></tr><tr><td>checkpoints/interfacegan_directions/</td><td>StyleGAN editing directions.</td></tr><tr><td>checkpoints/stylegan2_d_256.pth</td><td>Pre-trained StyleGAN discriminator.</td></tr><tr><td>checkpoints/model_ir_se50.pth</td><td>Pre-trained id-loss discriminator.</td></tr><tr><td>checkpoints/StyleHEAT_visual.pt</td><td>Pre-trained StyleHEAT model.</td></tr><tr><td>checkpoints/BFM</td><td>3DMM library. (Note the zip file should be unzipped to BFM/.)</td></tr><tr><td>checkpoints/Deep3D/epoch_20.pth</td><td>Pre-trained 3DMM extractor.</td></tr></tbody>
  </table></figure></details>

  **预训练文件说明**

- 如使用音频推演，请下载sadtalker[预训练模型](https://pan.baidu.com/s/1kb1BCPaLOWX1JJb9Czbn6w?pwd=sadt)和[离线包](https://pan.baidu.com/s/1P4fRgk9gaSutZnn8YW034Q?pwd=sadt),并根据sadtalker[说明](https://github.com/OpenTalker/SadTalker)放入对应的文件夹

### 模型推理

#### 视频重演

- **功能说明**

  该功能基于源**视频**生成**相同身份**的TalkingFace视频

- **参数说明**

  使用视频进行重演需要额外指定运行模式为**推理模式**并给出**源视频**。

  如需更改配置文件和输出路径，可通过相应参数修改。

  ```bash
  # 额外参数
  --run_mode=infer
  --video_source=./dataset/StyleHeat/videos/RD_Radio34_003_512.mp4
  (可选) --config ./talkingface/properties/model/StyleHeat/inference.yaml
  (可选) --output_dir=./dataset/StyleHeat/output
  ```

- **运行截图**

  <img src="http://image.lynxcc.top/image-20240130085501935.png" alt="image-20240130085501935" />

  <img src="http://image.lynxcc.top/image-20240130085547416.png" alt="image-20240130085547416" />

- **结果示例**

  左侧为原视频，右侧为模型生成视频

  <img src="http://image.lynxcc.top/image-20240130085935768.png" alt="image-20240130085935768" />

#### 图片重演

- **功能说明**

  该功能可用于**跨身份重演**，可使用**图片**模仿**源视频**生成TalkingFace视频。

- **参数说明**

  使用图片和视频进行重演需要额外指定运行模式为**推理模式**并给出**源视频**与**模仿者的面部图片**，要求图片长宽比例为1:1。

  如跨身份重演需给出`--cross_id`参数；如果需要对齐（裁剪）图像，请指定`--if_align`；如需提取目标视频的3dmm参数请指定`--if_extract`；如需更改配置文件和输出路径，可通过相应参数修改。

  ```bash
  # 额外参数
  --run_mode=infer
  --video_source=./dataset/StyleHeat/videos/RD_Radio34_003_512.mp4
  --image_source=./dataset/StyleHeat/images/100.jpg \
  (可选) --cross_id
  (可选) --if_extract
  (可选) --if_align
  (可选) --config ./talkingface/properties/model/StyleHeat/inference.yaml
  (可选) --output_dir=./dataset/StyleHeat/output
  ```

- **运行截图**

  <img src="http://image.lynxcc.top/image-20240130094424392.png" alt="image-20240130094424392" />

- **结果示例**

  左侧为原视频与原图片，右侧为模型生成视频

  <img src="http://image.lynxcc.top/image-20240130094952910.png" alt="image-20240130094952910" />

#### 音频重演

- **功能说明**

  该功能可用于**跨身份重演**，可使用**图片**和**音频**生成TalkingFace视频。

- **参数说明**

  使用图片和音频进行重演需要额外指定运行模式为**推理模式**并给出**源音频**与**面部图片**，要求图片长宽比例为1:1。

  如跨身份重演需给出`--cross_id`参数；如果需要对齐（裁剪）图像，请指定`--if_align`；如需提取目标视频的3dmm参数请指定`--if_extract`；如需更改配置文件和输出路径，可通过相应参数修改。

  ```bash
  # 额外参数
  --run_mode=infer
  --audio_path=./dataset/StyleHeat/audios/RD_Radio31_000.wav
  --image_source=./dataset/StyleHeat/images/100.jpg 
  (可选) --cross_id
  (可选) --if_extract
  (可选) --if_align
  (可选) --config ./talkingface/properties/model/StyleHeat/inference.yaml
  (可选) --output_dir=./dataset/StyleHeat/output
  ```

- **运行截图**

  <img src="http://image.lynxcc.top/image-20240130094424392.png" alt="image-20240130094424392" />

- **结果示例**

  左侧为原视频与原图片，右侧为模型生成视频

  <img src="http://image.lynxcc.top/image-20240130094952910.png" alt="image-20240130094952910" />

  > 注：推理中使用的示例视频可从[此处](https://drive.google.com/drive/folders/1-m47oPsa3kxjgK5eSJ8g8sHzG4zr2oRc?usp=sharing)下载

### 模型训练

模型训练分为两个阶段，要训练 VideoWarper，需使用VoxCelebA数据集；要训练整个框架还需HDTF数据集

#### VOX-StyleHeatWarpper

- **数据集准备**

  该训练使用vox数据集，但需对数据进行预处理，使视频裁剪到适合训练的大小并保证人脸居中。为了提高训练的速率，还需提前提取出视频的3dmm参数。最后将训练数据统一存入lmdb数据库中。

  - **数据集下载、裁剪与分割**

    我们使用[Video Preprocessing](https://github.com/AliaksandrSiarohin/video-preprocessing) 进行数据集的下载与预处理。我们对原仓库脚本做了一些修订使其符合本项目，该代码已包含在`talkingface/data/dataprocess/video-preprocessing` 路径中。

    预处理前，首先需要下载视频解释的文本文件，该预处理工具根据他下载所需的视频并处理成相应的格式。

    ```bash
    wget www.robots.ox.ac.uk/~vgg/data/voxceleb/data/vox2_test_txt.zip
    unzip vox1_test_txt.zip
    wget www.robots.ox.ac.uk/~vgg/data/voxceleb/data/vox2_dev_txt.zip
    unzip vox1_dev_txt.zip
    ```

    由于特殊格式的限制，预处理只能使用从**Youtube**上下载的视频。需要提前在环境中安装**ffmpeg**和**youtube-dl**，并按目录下requirements.txt安装需要的包。

    为了加速ffmpeg的处理速度，需要在环境中安装**Cuda**。如果无法使用Cuda，需要对代码中的命令进行修改。

    此外，需要face-alignment库，请从仓库下载该部分代码并安装到环境中。

    ```bash
    git clone https://github.com/1adrianb/face-alignment
    cd face-alignment
    pip install -r requirements.txt
    python setup.py install
    ```

    由于`youtube-dl`受法律限制停止更新，无法下载部分视频。因此我们使用**yt-dlp**进行代替，该工具已包含在预处理代码路径下。如需使用原来的`youtube-dl`，请将代码中的`yt-dlp`修改为`youtube-dl`。

    可以使用如下命令对视频进行预处理。`--format`可以指定存储格式（png/mp4）。PNG可以获得更好的IO性能，但需要更多的存储空间。

    ```bash
    python crop_vox.py --workers 40 --device_ids 0,1,2,3,4,5,6,7 --format .mp4 --dataset_version 2
    python crop_vox.py --workers 40 --device_ids 0,1,2,3,4,5,6,7 --format .mp4 --dataset_version 2 --data_range 10000-11252
    ```

  - **数据集参数提取与存储**

    进行完数据集的下载并裁剪分割后，需要提取视频的3dmm参数并将其存入lmdb数据库中。该部分代码已包含在`talkingface/data/dataprocess/PIRender` 路径中。可参照[PIRenderer](https://github.com/RenYurui/PIRender)完成该部分内容，但我们同样对该部分代码进行了部分修改，例如将keypoints参数保存在数据库中用于StyleHeat的训练。

    提取3dmm参数的过程使用了另外一个项目—— [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction)。下面是提取3dmm参数的过程：

    ---

    1. 按照他们的存储库的说明来构建 DeepFaceRecon 的环境。

    2. 将提供的脚本复制到文件夹中`Deep3DFaceRecon_pytorch`。

       ```
       cp scripts/face_recon_videos.py ./Deep3DFaceRecon_pytorch
       cp scripts/extract_kp_videos.py ./Deep3DFaceRecon_pytorch
       cp scripts/coeff_detector.py ./Deep3DFaceRecon_pytorch
       cp scripts/inference_options.py ./Deep3DFaceRecon_pytorch/options
       cd Deep3DFaceRecon_pytorch
       ```

    3. 从视频中提取面部标志。

       ```
       python extract_kp_videos.py \
       --input_dir path_to_viodes \
       --output_dir path_to_keypoint \
       --device_ids 0,1,2,3 \
       --workers 12
       ```

       ![image-20240130112349173](http://image.lynxcc.top/image-20240130112349173.png)

    4. 提取视频系数

       ```
       python face_recon_videos.py \
       --input_dir path_to_videos \
       --keypoint_dir path_to_keypoint \
       --output_dir output_dir \
       --inference_batch_size 100 \
       --name=model_name \
       --epoch=20 \
       --model facerecon
       ```

       <img src="http://image.lynxcc.top/image-20240130112239415.png" alt="image-20240130112239415" />

       ---

       提取后，会获得keypoints与3dmm系数两部分信息，文件夹格式应如下所示：

       ```bash
       ${DATASET_ROOT_FOLDER}
       └───path_to_videos
           └───train
               └───xxx.mp4
               └───xxx.mp4
               ...
           └───test
               └───xxx.mp4
               └───xxx.mp4
               ...
       └───path_to_3dmm_coeff
           └───train
               └───xxx.mat
               └───xxx.mat
               ...
           └───test
               └───xxx.mat
               └───xxx.mat
               ...
       ```

       最后将视频和 3DMM 参数保存在 lmdb 文件中。请运行以下代码来执行此操作:

       ```bash
       python scripts/prepare_vox_lmdb.py \
       --path path_to_videos \
       --coeff_3dmm_path path_to_3dmm_coeff \
       --out path_to_output_dir
       ```

       示例：

       <img src="http://image.lynxcc.top/image-20240130112059849.png" alt="image-20240130112059849" />

- **训练过程**

  如训练模型请使用如下的脚本

  ```bash
  python run_talkingface.py --model=StyleHeat --dataset=styleheat_vox
  ```

  可指定的额外参数如下：

  ```bash
  --runmode train
  --checkpoints_dir=./output
  --config talkingface/properties/model/StyleHeat/video_warper_trainer.yaml
  --name train_video_warper
  --single_gpu	
  ```

- **运行示例**

  <img src="http://image.lynxcc.top/image-20240130105956902.png" alt="image-20240130105956902" />

  在output中生成的日志文件如下：

  <img src="http://image.lynxcc.top/image-20240130110247353.png" alt="image-20240130110247353" />

- **训练结果**

  训练中会对模型进行评估，评估图片如下：

  <img src="http://image.lynxcc.top/image-20240130110437352.png" alt="image-20240130110437352" />

  随着训练epoch增加，生成的视频逐渐清晰，细节更加完整。

  epoch04:

  <img src="http://image.lynxcc.top/epoch_00002_iteration_000000024.jpg" alt="epoch_00002_iteration_000000024" style="zoom:50%;" />

  epoch18:

  <img src="http://image.lynxcc.top/epoch_00018_iteration_000000216.jpg" alt="epoch_00018_iteration_000000216" style="zoom: 50%;" />

#### HDTF-StyleHeat

- **数据集准备**

  该训练使用HDTF数据集并进行简单预处理

- **训练过程**

  如训练模型请使用如下的脚本

  ```bash
  python run_talkingface.py --model=StyleHeat --dataset=styleheat_vox
  ```

  可指定的额外参数如下：

  ```bash
  --runmode train
  --checkpoints_dir=./output
  --config talkingface/properties/model/StyleHeat/video_styleheat_trainer.yaml
  --name train_video_styleheat
  --single_gpu	
  ```

- **运行示例**

  <img src="http://image.lynxcc.top/image-20240130111541161.png" alt="image-20240130111541161" />

## 模型实现

- 由于增加了推理部分的内容，因此对quick_start.py进行了部分修改，增加了一个StyleHeat模型的特殊处理。
- 由于模型拥有多个网络，且训练数据集存在数据库中而不是使用文件列表。因此训练过程直接重写了trainer.py中的fit函数，而获取数据集部分返回的则是加载后数据集列表的句柄。模型整体框架仍使用talkingface-toolkit结构