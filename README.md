# PHADTF

## 成员分工

孙宇轩：分析PHADTF论文和源代码并添加注释，重构了GAN模型的model、dataset和trainer三部分以及最终quikstart部分，将模型的各部分进行连接，规范整体框架格式，对模型进行的调试与修改工作。\
张祖铭：分析PHADTF论文和源代码并添加注释，负责backgroundmatch部分的代码迁移重构，对模型进行的调试与修改工作，编写说明文档的colab运行部分。&#x20;

肖冰馨：分析PHADTF论文和源代码并添加注释，重构了LSTM模型的model、和trainer两部分，参与模型调试工作，编写说明文档。&#x20;

范禹坤：分析PHADTF论文和源代码并添加注释，撰写Deep3dFaceReconstruction.py和调试文件test.py，参与模型调试工作，编写说明文档。&#x20;

梅若曦：分析PHADTF论文和源代码并添加注释，重构了LSTM模型的model、和trainer两部分，参与模型调试工作，编写说明文档部分。

## 原论文阅读笔记

### 论文背景以及遇到的问题

视觉和听觉模态是人与人或人机交互中两个重要的感觉通道。这两种模态中的信息是强相关的在真实场景中，自然的头部运动在高质量的交流中起着重要的作用，现实世界的说话往往伴随着自然的头部运动，而目前大部分的的说话脸视频的生成方法都是只考虑固定头部姿势的面部动画。

这并不是一个新的问题，相关研究人员想出从语音中推断头部姿势这一方法，虽然已经观察到语音和头部姿势之间存在一些可测量的相关性，但从语音中预测头部运动仍然是一个具有挑战性的问题。

### 相关研究进展

#### 动态face生成

现有的说话脸视频生成方法大致分为两类：视频驱动和音频驱动。前者将表情、头部姿势从驱动帧转移到目标演员的面部图像上；后者则是采用一段音频和任意一张人脸图像作为输入，或使用编码器-解码器 CNN 模型生成会说话的人脸视频，或通过学习联合视听表示来生成说话脸。

但是上面提到的方法都是在2D平面内，在其谈话过程中，头部姿势几乎是固定的，很难仅仅使用 2D 信息来自然地建模姿势的变化。

#### 3D人脸重建

因为上面2D平面内生成说话脸的方法有很大缺陷，作者团队则引入了3D几何信息，同时对个性化的头部姿势、表情和嘴唇同步进行建模。

这一领域已经提出了大量的方法，作者团队选择了使用CNN学习人脸图像到3DMM参数的模型。

#### GANs和记忆网络

目前，生成对抗网络（GANs）已经成功应用于多个计算机视觉问题，后来又扩展到视频到视频的合成和面部动画及纹理合成领域。

记忆网络则是一种利用用外部记忆增强神经网络的方案。它已被应用于问答系统、图像字幕和图像着色。在原论文中，作者使用增强记忆网络的GAN来微调渲染帧到任意人的真实帧。

### 作者提到的解决方案的关键

#### 为了在说话源人的输入音频信号时输出具有个性化头部 姿势的目标人的高质量合成视频，作者团队采用了3D面部动画，并将其重新渲染成视频帧。

然后，他们提出了一种新的记忆增强 GAN 模块，该模块可以根据目标人物的身份特征，将粗糙的渲染帧细化为平滑过渡的逼真帧。

这是第一个可以将任意源音频信号转换成任意目标头部姿势的面部说话的系统。

作者提出的解决方法分为两个阶段。第一阶段使用LRW视频数据集训练从音频语音到面部表情和常见头部姿势的一般映射。获得具有个性化头部姿势的3D面部动画。

第二阶段利用从输入视频中获得的纹理和光照信息，将3D面部动画渲染成视频帧，并通过新的内存增强GAN模块处理各种身份，生成高质量的帧。

### 具体实验的设计

作者在PyTorch中实现了其方法，在其模型中，LSTM和记忆增强GAN涉及两个训练步骤：LRW视频数据集训练的一般映射和微调一般映射以学习个性化的说话行为。

他们从Youtube上手机了15个真实世界中单人的说话视频，每个视频中，使用其前12秒作为训练数据。将网络先在一般映射（一般数据集）中训练，然后在微调的个性化映射（针对特定的人）中进行微调。

接着作者团队对模型中两个阶段的重要性进行了评估，并将模型中特有的一般映射以及微调的个性化映射与先前的其他方法进行了对比。

先将个性化映射与YouSaidThat、DAVS、ATVG三种基于2D的方法进行了对比，通过一些定性的结果图片表明作者的方法在所有的三个标准中都取得的更好的结果。

再进一步将通用映射与代表性的音频驱动方法进行了比较，将不同方法生成的结果与ground-truth视频进行了对比，得出了一些定量的比较结果，表明作者的方法具有最佳的PSNR值。

在实验的最后，为了客观地评价个性化头部姿态的质量，作者提出了一种新的度量 HS 来衡量生成的视频与真实视频之间的头部姿态相似度。HS越大，表明头部姿势的相似度越高。

### 论文的贡献以及未来可以继续研究的方向

这篇论文提出了一个深度神经网络模型，可以生成目标人的高质量说话视频，该模型使用了3D面部动画来弥补了视听驱动的头部姿势学习和逼真的说话人脸视频的生成之间的差距，同时还使用内存增强GAN模块将渲染的帧微调成逼真的帧。最终的实验结果表明，这一模型能够生成具有个性化头部姿势的高质量说话头像视频。

在论文的最后，作者提到，当帧数较低时，生成的视频质量会较低，可能是因为在个性化映射中，使用的是低频反照率，而将低频反照率微调成逼真的反照率需要更多的帧数。但是在一般映射中，则可以使用高频反照率，因此，在通用映射中只用一帧可能会比在个性化映射中使用几帧来微调反照率产生更加逼真的效果。

## 环境依赖

运行代码需要如下环境：

python3.8

GPU

MATLAB

以及运行如下指令来安装依赖环境：

`pip install -r requirements.txt`

## 在colab上运行

我们组提供一个demo可以成功在colab上运行。论文原文所给的demo由于时间过于古老因此安装包并不适配，我们修改安装包安装方式，实现运行。

### 准备工作

首先克隆代码到notebook。

```notebook-python
!git clone https://github.com/yiranran/Audio-driven-TalkingFace-HeadPose.git
```

代码克隆之后进入该文件夹执行如下代码进行安装环境：

```notebook-python
pip install -r requirements_colab.txt
```

但是不出意外应该会报错，因为默认情况下colab使用最新的稳定版本python，比如现在（2024年1月29日）所用为3.10.0。但是论文原文和所用的依赖包并不支持这个版本，因此需要在执行上述命令之前安装python3.8。输入如下指令：

    %%bash
    MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
    MINICONDA_PREFIX=/usr/local
    wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
    chmod +x $MINICONDA_INSTALLER_SCRIPT
    ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX

    conda install --channel defaults conda python=3.8 --yes
    conda update --channel defaults --all --yes

<!---->

    import sys
    sys.path.append("/usr/local/lib/python3.8/site-packages")

之后输入命令行显示python当前版本，可以显示如图：

之后再安装依赖环境，显示如下：

![](2_md_files/9f0267b0-bebf-11ee-8e17-d3ef749341c8.jpeg?v=1\&type=image)

可以看到安装都已完成。

在云端时无法使用MATLAB的因此需要做替代：

```notebook-python
!apt install octave liboctave-dev
```

使用如上的安装代替MATLAB。

![](2_md_files/17fb7da0-bec0-11ee-8e17-d3ef749341c8.jpeg?v=1\&type=image)

#### 下载和配置预训练模型

预训练模型原文中给了两种下载方式，其一是使用百度网盘，其二是使用google网盘，由于我们使用的是colab因此需要使用google网盘。由于华为云无法使用网盘，所以只能下载百度网盘之后手动上传到OBS桶中，我们就不这么做了。

![](2_md_files/91393db0-bec0-11ee-8e17-d3ef749341c8.jpeg?v=1\&type=image)

#### 下载用于 3D 面部重建的面部模型

面部重建也需要下载预训练模型，在3D人脸重建中会用到。

*   从 <https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model> 下载 Basel Face Model，并将 01\_MorphableModel.mat 复制到 Deep3DFaceReconstruction/BFM 文件夹

*   从Guo等人的CoarseData中下载Expression Basis，并将Exp\_Pca.bin复制到Deep3DFaceReconstruction/BFM文件夹中（[Juyong/3DFace: This repository contains the dataset including the pair of 2D face image and its corresponding 3D face geometry model. (github.com)](https://github.com/Juyong/3DFace)）

### 对目标人物视频进行微调

==这里的微调表示对预训练模型进行微调，也就是重新训练预训练模型。==

准备一个满足以下条件的会说话的面部视频：

1.  包含一个人

2.  25 fps

3.  长度超过 12 秒

4.  没有较大的身体平移（例如从屏幕的左侧移动到右侧）

将视频重命名为 \[person\_id].mp4（例如 1.mp4）并复制到 Data 子文件夹。这段视屏将作为目标人物，要对其进行克隆操作。

处理完mp4视频之后我们就可以进行抽帧或者

#### 抽帧操作

```notebook-python
!cd Data/; python extract_frame1.py 31.mp4
```

在该目录下进行抽帧操作，抽帧之后会保存。抽帧之后方便与后续操作。命令行显示如下：

![](2_md_files/064cb830-bec5-11ee-8e17-d3ef749341c8.jpeg?v=1\&type=image)

#### 在后续处理中可能会冲突的安装包

后续操作过程可能会有安装包冲突，因此需要重新安装安装包以满足后续对于音频信息的处理。

```notebook-python
!pip list | grep tensorflow
```

```notebook-python
!pip uninstall tensorflow tensorflow-gpu
```

```notebook-python
!pip install tensorflow-gpu==1.14.0
```

在执行完这些指令之后 我们就可以构建主体模型框架了。

#### 构建 tf\_mesh\_renderer

现在构建talkingface网格渲染，这部分需要用到MATLAB和c++代码，很有可能无法运行成功，需要自行根据自身电脑的情况修改特点代码。

输入如下指令，将代码复制到指定地方，完成构建。

```notebook-python
!cp /usr/local/lib/python3.6/dist-packages/tensorflow/libtensorflow_framework.so.1 /usr/lib/
!cd /usr/lib/ && ln -s libtensorflow_framework.so.1 libtensorflow_framework.so
!cd Deep3DFaceReconstruction/tf_mesh_renderer/mesh_renderer/kernels/;\
  g++ -std=c++11 -shared rasterize_triangles_grad.cc rasterize_triangles_op.cc rasterize_triangles_impl.cc rasterize_triangles_impl.h -o rasterize_triangles_kernel.so -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I /usr/local/lib/python3.6/dist-packages/tensorflow/include -I /usr/local/lib/python3.6/dist-packages/tensorflow/include/external/nsync/public -L /usr/local/lib/python3.6/dist-packages/tensorflow -ltensorflow_framework -O2
```

#### 使用 pycat 和 %%writefile 编辑 rasterize\_triangles.py

因为在colab运行环境与本地运行并不相同，因此需要修改部分代码以适应环境。

代码如下：

```notebook-python
%%writefile Deep3DFaceReconstruction/tf_mesh_renderer/mesh_renderer/rasterize_triangles.py
# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable triangle rasterizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from . import camera_utils

rasterize_triangles_module = tf.load_op_library(
    #os.path.join(os.environ['TEST_SRCDIR'],
    os.path.join('/content/Audio-driven-TalkingFace-HeadPose/Deep3DFaceReconstruction',
    'tf_mesh_renderer/mesh_renderer/kernels/rasterize_triangles_kernel.so'))


def rasterize(world_space_vertices, attributes, triangles, camera_matrices,
              image_width, image_height, background_value):
  """Rasterizes a mesh and computes interpolated vertex attributes.

  Applies projection matrices and then calls rasterize_clip_space().

  Args:
    world_space_vertices: 3-D float32 tensor of xyz positions with shape
      [batch_size, vertex_count, 3].
    attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
      attribute_count]. Each vertex attribute is interpolated across the
      triangle using barycentric interpolation.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
      should contain vertex indices describing a triangle such that the
      triangle's normal points toward the viewer if the forward order of the
      triplet defines a clockwise winding of the vertices. Gradients with
      respect to this tensor are not available.
    camera_matrices: 3-D float tensor with shape [batch_size, 4, 4] containing
      model-view-perspective projection matrices.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    background_value: a 1-D float32 tensor with shape [attribute_count]. Pixels
      that lie outside all triangles take this value.

  Returns:
    A 4-D float32 tensor with shape [batch_size, image_height, image_width,
    attribute_count], containing the interpolated vertex attributes at
    each pixel.

  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  clip_space_vertices = camera_utils.transform_homogeneous(
      camera_matrices, world_space_vertices)
  return rasterize_clip_space(clip_space_vertices, attributes, triangles,
                              image_width, image_height, background_value)


def rasterize_clip_space(clip_space_vertices, attributes, triangles,
                         image_width, image_height, background_value):
  """Rasterizes the input mesh expressed in clip-space (xyzw) coordinates.

  Interpolates vertex attributes using perspective-correct interpolation and
  clips triangles that lie outside the viewing frustum.

  Args:
    clip_space_vertices: 3-D float32 tensor of homogenous vertices (xyzw) with
      shape [batch_size, vertex_count, 4].
    attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
      attribute_count]. Each vertex attribute is interpolated across the
      triangle using barycentric interpolation.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
      should contain vertex indices describing a triangle such that the
      triangle's normal points toward the viewer if the forward order of the
      triplet defines a clockwise winding of the vertices. Gradients with
      respect to this tensor are not available.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    background_value: a 1-D float32 tensor with shape [attribute_count]. Pixels
      that lie outside all triangles take this value.

  Returns:
    A 4-D float32 tensor with shape [batch_size, image_height, image_width,
    attribute_count], containing the interpolated vertex attributes at
    each pixel.

  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  if not image_width > 0:
    raise ValueError('Image width must be > 0.')
  if not image_height > 0:
    raise ValueError('Image height must be > 0.')
  if len(clip_space_vertices.shape) != 3:
    raise ValueError('The vertex buffer must be 3D.')

  vertex_count = clip_space_vertices.shape[1].value

  batch_size = tf.shape(clip_space_vertices)[0]
  
  per_image_barycentric_coordinates = tf.TensorArray(dtype=tf.float32,
    size=batch_size)
  per_image_vertex_ids = tf.TensorArray(dtype=tf.int32, size=batch_size)

  def batch_loop_condition(b, *args):
    return b < batch_size

  def batch_loop_iteration(b, per_image_barycentric_coordinates,
    per_image_vertex_ids):
    barycentric_coords, triangle_ids, _ = (
        rasterize_triangles_module.rasterize_triangles(
            clip_space_vertices[b, :, :], triangles, image_width,
            image_height))
    per_image_barycentric_coordinates = \
      per_image_barycentric_coordinates.write(
        b, tf.reshape(barycentric_coords, [-1, 3]))

    vertex_ids = tf.gather(triangles, tf.reshape(triangle_ids, [-1]))
    reindexed_ids = tf.add(vertex_ids, b * clip_space_vertices.shape[1].value)
    per_image_vertex_ids = per_image_vertex_ids.write(b, reindexed_ids)

    return b+1, per_image_barycentric_coordinates, per_image_vertex_ids

  b = tf.constant(0)
  _, per_image_barycentric_coordinates, per_image_vertex_ids = tf.while_loop(
    batch_loop_condition, batch_loop_iteration,
    [b, per_image_barycentric_coordinates, per_image_vertex_ids])

  barycentric_coordinates = tf.reshape(
    per_image_barycentric_coordinates.stack(), [-1, 3])
  vertex_ids = tf.reshape(per_image_vertex_ids.stack(), [-1, 3])

  # Indexes with each pixel's clip-space triangle's extrema (the pixel's
  # 'corner points') ids to get the relevant properties for deferred shading.
  flattened_vertex_attributes = tf.reshape(attributes,
                                           [batch_size * vertex_count, -1])
  corner_attributes = tf.gather(flattened_vertex_attributes, vertex_ids)

  # Computes the pixel attributes by interpolating the known attributes at the
  # corner points of the triangle interpolated with the barycentric coordinates.
  weighted_vertex_attributes = tf.multiply(
      corner_attributes, tf.expand_dims(barycentric_coordinates, axis=2))
  summed_attributes = tf.reduce_sum(weighted_vertex_attributes, axis=1)
  attribute_images = tf.reshape(summed_attributes,
                                [batch_size, image_height, image_width, -1])

  # Barycentric coordinates should approximately sum to one where there is
  # rendered geometry, but be exactly zero where there is not.
  alphas = tf.clip_by_value(
      tf.reduce_sum(2.0 * barycentric_coordinates, axis=1), 0.0, 1.0)
  alphas = tf.reshape(alphas, [batch_size, image_height, image_width, 1])

  attributes_with_background = (
      alphas * attribute_images + (1.0 - alphas) * background_value)

  return attributes_with_background


@tf.RegisterGradient('RasterizeTriangles')
def _rasterize_triangles_grad(op, df_dbarys, df_dids, df_dz):
  # Gradients are only supported for barycentric coordinates. Gradients for the
  # z-buffer are not currently implemented. If you need gradients w.r.t. z,
  # include z as a vertex attribute when calling rasterize_triangles.
  del df_dids, df_dz
  return rasterize_triangles_module.rasterize_triangles_grad(
      op.inputs[0], op.inputs[1], op.outputs[0], op.outputs[1], df_dbarys,
      op.get_attr('image_width'), op.get_attr('image_height')), None
```

最后需要复写源代码：

```notebook-python
Overwriting Deep3DFaceReconstruction/tf_mesh_renderer/mesh_renderer/rasterize_triangles.py
```

同理继续修改代码：

```notebook-python
%%writefile Audio/code/mesh_renderer/rasterize_triangles.py
# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable triangle rasterizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from . import camera_utils

rasterize_triangles_module = tf.load_op_library(
    #os.path.join(os.environ['TEST_SRCDIR'],
    os.path.join('/content/Audio-driven-TalkingFace-HeadPose/Deep3DFaceReconstruction',
    'tf_mesh_renderer/mesh_renderer/kernels/rasterize_triangles_kernel.so'))


def rasterize(world_space_vertices, attributes, triangles, camera_matrices,
              image_width, image_height, background_value):
  """Rasterizes a mesh and computes interpolated vertex attributes.

  Applies projection matrices and then calls rasterize_clip_space().

  Args:
    world_space_vertices: 3-D float32 tensor of xyz positions with shape
      [batch_size, vertex_count, 3].
    attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
      attribute_count]. Each vertex attribute is interpolated across the
      triangle using barycentric interpolation.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
      should contain vertex indices describing a triangle such that the
      triangle's normal points toward the viewer if the forward order of the
      triplet defines a clockwise winding of the vertices. Gradients with
      respect to this tensor are not available.
    camera_matrices: 3-D float tensor with shape [batch_size, 4, 4] containing
      model-view-perspective projection matrices.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    background_value: a 1-D float32 tensor with shape [attribute_count]. Pixels
      that lie outside all triangles take this value.

  Returns:
    A 4-D float32 tensor with shape [batch_size, image_height, image_width,
    attribute_count], containing the interpolated vertex attributes at
    each pixel.

  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  clip_space_vertices = camera_utils.transform_homogeneous(
      camera_matrices, world_space_vertices)
  return rasterize_clip_space(clip_space_vertices, attributes, triangles,
                              image_width, image_height, background_value)


def rasterize_clip_space(clip_space_vertices, attributes, triangles,
                         image_width, image_height, background_value):
  """Rasterizes the input mesh expressed in clip-space (xyzw) coordinates.

  Interpolates vertex attributes using perspective-correct interpolation and
  clips triangles that lie outside the viewing frustum.

  Args:
    clip_space_vertices: 3-D float32 tensor of homogenous vertices (xyzw) with
      shape [batch_size, vertex_count, 4].
    attributes: 3-D float32 tensor with shape [batch_size, vertex_count,
      attribute_count]. Each vertex attribute is interpolated across the
      triangle using barycentric interpolation.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
      should contain vertex indices describing a triangle such that the
      triangle's normal points toward the viewer if the forward order of the
      triplet defines a clockwise winding of the vertices. Gradients with
      respect to this tensor are not available.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    background_value: a 1-D float32 tensor with shape [attribute_count]. Pixels
      that lie outside all triangles take this value.

  Returns:
    A 4-D float32 tensor with shape [batch_size, image_height, image_width,
    attribute_count], containing the interpolated vertex attributes at
    each pixel.

  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  if not image_width > 0:
    raise ValueError('Image width must be > 0.')
  if not image_height > 0:
    raise ValueError('Image height must be > 0.')
  if len(clip_space_vertices.shape) != 3:
    raise ValueError('The vertex buffer must be 3D.')

  vertex_count = clip_space_vertices.shape[1].value

  batch_size = tf.shape(clip_space_vertices)[0]
  
  per_image_barycentric_coordinates = tf.TensorArray(dtype=tf.float32,
    size=batch_size)
  per_image_vertex_ids = tf.TensorArray(dtype=tf.int32, size=batch_size)

  def batch_loop_condition(b, *args):
    return b < batch_size

  def batch_loop_iteration(b, per_image_barycentric_coordinates,
    per_image_vertex_ids):
    barycentric_coords, triangle_ids, _ = (
        rasterize_triangles_module.rasterize_triangles(
            clip_space_vertices[b, :, :], triangles, image_width,
            image_height))
    per_image_barycentric_coordinates = \
      per_image_barycentric_coordinates.write(
        b, tf.reshape(barycentric_coords, [-1, 3]))

    vertex_ids = tf.gather(triangles, tf.reshape(triangle_ids, [-1]))
    reindexed_ids = tf.add(vertex_ids, b * clip_space_vertices.shape[1].value)
    per_image_vertex_ids = per_image_vertex_ids.write(b, reindexed_ids)

    return b+1, per_image_barycentric_coordinates, per_image_vertex_ids

  b = tf.constant(0)
  _, per_image_barycentric_coordinates, per_image_vertex_ids = tf.while_loop(
    batch_loop_condition, batch_loop_iteration,
    [b, per_image_barycentric_coordinates, per_image_vertex_ids])

  barycentric_coordinates = tf.reshape(
    per_image_barycentric_coordinates.stack(), [-1, 3])
  vertex_ids = tf.reshape(per_image_vertex_ids.stack(), [-1, 3])

  # Indexes with each pixel's clip-space triangle's extrema (the pixel's
  # 'corner points') ids to get the relevant properties for deferred shading.
  flattened_vertex_attributes = tf.reshape(attributes,
                                           [batch_size * vertex_count, -1])
  corner_attributes = tf.gather(flattened_vertex_attributes, vertex_ids)

  # Computes the pixel attributes by interpolating the known attributes at the
  # corner points of the triangle interpolated with the barycentric coordinates.
  weighted_vertex_attributes = tf.multiply(
      corner_attributes, tf.expand_dims(barycentric_coordinates, axis=2))
  summed_attributes = tf.reduce_sum(weighted_vertex_attributes, axis=1)
  attribute_images = tf.reshape(summed_attributes,
                                [batch_size, image_height, image_width, -1])

  # Barycentric coordinates should approximately sum to one where there is
  # rendered geometry, but be exactly zero where there is not.
  alphas = tf.clip_by_value(
      tf.reduce_sum(2.0 * barycentric_coordinates, axis=1), 0.0, 1.0)
  alphas = tf.reshape(alphas, [batch_size, image_height, image_width, 1])

  attributes_with_background = (
      alphas * attribute_images + (1.0 - alphas) * background_value)

  return attributes_with_background


@tf.RegisterGradient('RasterizeTriangles')
def _rasterize_triangles_grad(op, df_dbarys, df_dids, df_dz):
  # Gradients are only supported for barycentric coordinates. Gradients for the
  # z-buffer are not currently implemented. If you need gradients w.r.t. z,
  # include z as a vertex attribute when calling rasterize_triangles.
  del df_dids, df_dz
  return rasterize_triangles_module.rasterize_triangles_grad(
      op.inputs[0], op.inputs[1], op.outputs[0], op.outputs[1], df_dbarys,
      op.get_attr('image_width'), op.get_attr('image_height')), None
```

Overwriting Audio/code/mesh\_renderer/rasterize\_triangles.py继续重写

#### 运行 3D 人脸重建

输入指令运行重建过程

```notebook-python
!cd Deep3DFaceReconstruction/; CUDA_VISIBLE_DEVICES=0 python demo_19news.py ../Data/31
```

#### 微调音频网络

```notebook-python
!cd Audio/code/; python train_19news_1.py 31 0
```

这里就是运行的主要内容了，运行过程如图：

![](2_md_files/470fc0b0-beca-11ee-8e17-d3ef749341c8.jpeg?v=1\&type=image)

可以看到损失函数和运行次数。epoch和step

#### 训练对抗神经网络（GAN）

这里也是对预训练模型进行精度训练，训练之后我们我们就可以生成视频了。

![](2_md_files/8c5891b0-beca-11ee-8e17-d3ef749341c8.jpeg?v=1\&type=image)

### 进行测试（评估）

#### 在我们自己上传的人物视频中测试

使用音频 03Fsi1831.wav 对 31 人进行测试

运行如下指令进行测试

```notebook-python
!cd Audio/code/; python test_personalized2.py 03Fsi1831 31 0
```

结果保存到 ..[/results/atcnet\_pose0\_con3/31/03Fsi1831\_99/31\_03Fsi1831wav\_results\_full9.mov](https://colab.research.google.com/drive/1gqcqTSAGAyj48n0fmApvSPG_43BzKP37#) 和 ..[/results/atcnet\_pose0\_con3/31/03Fsi1831\_99/31\_03Fsi1831wav\_results\_transbigbg.mov](https://colab.research.google.com/drive/1gqcqTSAGAyj48n0fmApvSPG_43BzKP37#)两个地方

#### 展示结果视频

也就是我们自己生成的人物说话模型。

```notebook-python
from IPython.display import HTML
from base64 import b64encode

video_path = 'Audio/results/atcnet_pose0_con3/31/03Fsi1831_99/31_03Fsi1831wav_results_transbigbg.mov'

mp4 = open(video_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)
```

下面是运行最后的结果

![](2_md_files/61927210-becb-11ee-b5a9-f503f59f5266.jpeg?v=1\&type=image)

运行方法：

## 提交的代码使用方法：

1.  准备数据人像说话视频，mp4格式解压到目录dataset/PHADTF/data下，要求25fps，可以通过ffmpeg -i xxx.mp4 -r 25 xxx.mp4来达到

2.  下载[预训练权重文件](https://drive.google.com/file/d/17xMNjNEsM0DhS9SKDdUQue0wlbz4Ww9o)，解压到目录checkpoints/PHADTF下

3.  运行 run\_talkingface.py --model PHADTF\_LSTM --dataset PHADTF

4.  运行utils\PHADTF\_backgroud\_blending里的文件，需要安装MATLAB

5.  运行 run\_talkingface.py --model PHADTF\_GAN --dataset PHADTF

代码说明：

### 因为原仓库里的代码，就是按照上述论文中的结构分块写成的。不同段落之间代码底层结构、环境依赖和模型结构完全不同，中间还需要调用外部文件。所以模型分成预处理头像和处理语言的LSTM一半，和合并参数、产生视频的GAN一半

### 模型没有提供真正的从零开始训练，只有根据输入的目标视频fine-tune网络的部分

models部分下，PHADTF\_LSTMmodel作为前半段训练的入口，同时调用PHADTF\_\_backgroundmatch.py计算脸部模型；PHADTF\_GAN作为后半部分训练的入口，同时调用PHADTF\_arcface下的内容进行参数合并
