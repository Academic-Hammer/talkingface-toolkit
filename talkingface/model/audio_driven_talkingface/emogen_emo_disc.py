from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from abstract_talkingface import AbstractTalkingFace  # 假设这是AbstractTalkingFace的导入路径

class DISCEMO(AbstractTalkingFace):
    def __init__(self, debug=False):
        super(DISCEMO, self).__init__()
        self.debug = debug
        self.drp_rate = 0

        # 定义卷积层
        self.filters = [(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2), (512, 3, 2)]
        prev_filters = 3
        self.conv_layers = nn.ModuleList()
        for num_filters, filter_size, stride in self.filters:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(prev_filters, num_filters, kernel_size=filter_size, stride=stride, padding=filter_size//2),
                    nn.LeakyReLU(0.3)
                )
            )
            prev_filters = num_filters

        # 线性层和 RNN
        self.projector = nn.Sequential(
            nn.Linear(4608, 2048),
            nn.LeakyReLU(0.3),
            nn.Linear(2048, 512)
        )
        self.rnn_1 = nn.LSTM(512, 512, 1, bidirectional=False, batch_first=True)
        self.cls = nn.Sequential(
            nn.Linear(512, 6)
        )

        # 优化器和学习率调度器
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-06, betas=(0.5, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, 150, gamma=0.1, last_epoch=-1)

    def forward(self, video):
        x = video
        n, c, t, w, h = x.size()
        x = x.view(t * n, c, w, h)

        # 逐层应用卷积
        for conv in self.conv_layers:
            x = conv(x)

        # 将输出重新塑形并通过后续层
        h = x.view(n, t, -1)
        h = self.projector(h)
        h, _ = self.rnn_1(h)

        # 分类输出
        h_class = self.cls(h[:, -1, :])
        return h_class
    
    def calculate_loss(self, interaction):
        video = interaction['input']
        target = interaction['target']

        output = self.forward(video)
        loss = self.loss_func(output, target)
        return {'loss': loss}

    def predict(self, interaction):
        video = interaction['input']
        with torch.no_grad():
            output = self.forward(video)
        return output

    def generate_batch(self):
        # 实现批量生成数据的方法
        raise NotImplementedError

    # 其他方法可以根据需要保持不变或进行修改
