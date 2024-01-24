论文复现-LiveSpeechPortraits

**详情请查看文档"语音识别-论文复现-LiveSpeechPortraits.docx"以及“演示视频.mkv”**

完成的功能：复现论文，并验证。

训练：无，因为训练部分的代码涉及到论文作者所在公司的私有框架，由于版权问题不允许发布公开，另外在论文和仓库中作者也没有指明所需的数据集。

验证截图：![验证截图](.\验证截图.png)

验证说明：如果需要验证其他音频或其他说话人，请修改talkingface/properties/model处的live_speech_portraits.yaml，除model_params处的APC的ckp_path无需修改外，其他路径请都修改为你所需要指定的音频和说话人，这些数据均存放在checkpoints/live_speech_portraits内

所使用的依赖：

absl-py==2.1.0
aiosignal==1.3.1
albumentations==0.5.2
attrs==23.2.0
cachetools==5.3.2
click==8.1.7
colorlog==6.7.0
dominate==2.9.1
filelock==3.13.1
fonttools==4.25.0
frozenlist==1.4.1
google-auth==2.26.2
google-auth-oauthlib==0.4.6
grpcio==1.60.0
h5py==3.10.0
imageio==2.33.1
imgaug==0.4.0
Jinja2==3.1.3
jsonschema==4.21.0
jsonschema-specifications==2023.12.1
librosa==0.7.0
llvmlite==0.31.0
Markdown==3.5.2
MarkupSafe==2.1.4
mkl-service==2.4.0
mpmath==1.3.0
munkres==1.1.4
networkx==3.1
numba==0.48.0
numpy==1.20.3
oauthlib==3.2.2
opencv-python==3.4.9.33
opencv-python-headless==4.9.0.80
pandas==1.3.4
pkgutil_resolve_name==1.3.10
protobuf==3.19.0
pyasn1==0.5.1
pyasn1-modules==0.3.0
python-speech-features==0.6
pytz==2023.3.post1
PyWavelets==1.4.1
PyYAML==6.0.1
ray==2.6.3
referencing==0.32.1
requests-oauthlib==1.3.1
resampy==0.3.1
rpds-py==0.17.1
rsa==4.9
scikit-image==0.16.2
scipy==1.10.1
shapely==2.0.2
sympy==1.12
tensorboard==2.7.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
texttable==1.7.0
torch==1.13.1
torchaudio==0.13.1
torchvision==0.14.1
tqdm==4.66.1
tzdata==2023.4
Werkzeug==3.0.1

成员分工：晏永磊完成了所有工作，因为这个组只有他一人。

