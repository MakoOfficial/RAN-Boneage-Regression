import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

def get_ResNet():
    """获得主干网络ResNet50"""
    model = resnet50(pretrained=True)
    # 设置模型的输出通道,fc为ResNet中的最后一层，它的in_features即为输出的类别，就是输出通道，为2048
    output_channels = model.fc.in_features
    #   将网络中的所有子网络放入sequential，然后除去ResNet中最后的池化层和线性层，只保留了主干网络和前面的一些网络
    #   list(model.children())[:-2]的输出如下
    # [Conv2d(3, 64),
    # BatchNorm2d(64),
    # ReLU(),
    # MaxPool2d(kernel_size=3, stride=2, padding=1),
    # Sequential(),
    # Sequential(),
    # Sequential(),
    # Sequential()]
    # 计划就是在sequential之间穿插自制的MMCA模块
    ##
    model = list(model.children())[:-2]
    return model, output_channels

class RAm(nn.Module):
    "Rich Attention module"
    def __init__(self, output_channels, M) -> None:
        super().__init__()
        self.M = M
        self.output_channels = output_channels
        self.attention_generate_layer = nn.Sequential(
            nn.Conv2d(output_channels, M, kernel_size=1),
            nn.ReLU()
        )
        self.diversity = nn.Linear(output_channels, M)

    def generate_vector(self, atten_map, feature_map):
        # print(f"atten_map shape:{atten_map.shape}, featrue_map shape:{feature_map.shape}")
        return torch.squeeze(F.adaptive_avg_pool2d(atten_map*feature_map, 1))

    def forward(self, feature_map):
        attn_map = self.attention_generate_layer(feature_map)
        v = torch.zeros([attn_map.shape[0], self.M, self.output_channels], dtype=attn_map.dtype, device=attn_map.device)
        for i in range(self.M):
            v[:,i] = self.generate_vector(attn_map[:, i].unsqueeze(dim=1), feature_map)
        P = torch.zeros([feature_map.shape[0], self.M, self.M], dtype=v.dtype, device=v.device)
        for i in range(self.M):
            P[:, i] = self.diversity(v[:, i])

        return P, v

class RA_Net(nn.Module):
    "Rich Attention Net"
    def __init__(self, backbone, output_channels, M) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.M = M
        self.backbone = nn.Sequential(*backbone)
        self.RAm = RAm(output_channels, M)
        self.classifer = nn.Sequential(
            nn.Linear(output_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, image, ifTest):
        feature_map = self.backbone(image)

        x = F.adaptive_avg_pool2d(feature_map, 1)
        x = torch.squeeze(x)
        x = x.view(-1, self.output_channels)
        y_hat = self.classifer(x)

        if ifTest:
            return y_hat

        P, v = self.RAm(feature_map)
        y_RA = torch.zeros([v.shape[0], self.M], dtype=v.dtype,device=v.device)
        for i in range(self.M):
            y_RA[:, i] = self.classifer(v[:, i]).squeeze()
        return y_hat, y_RA, P

    # 加入微调函数
    def fine_tune(self, need_fine_tune = True):
        self.train(need_fine_tune)

class myres(nn.Module):
    """主模型MMANet的在输入到GA前的部分"""
    # 不在类内定义主干网络是因为怕梯度损失吗
    def __init__(self, backbone, out_channels) -> None:
        super().__init__()
        # self.resnet50 = get_ResNet()
        # 共有四块MMCA，所以这里分成四块来写，每块的主干部分和MMCA分开
        # 注意点：resnet总共四个sequential，输出通道分别是256, 512, 1024, 2048，这也确定MMCA的输入通道，但经过四层后高宽除以32
        # ResNet的前五层分别为：线性层conv2d，bn，ReLU，maxpooling，和第一个sequential
        self.out_channels = out_channels
        self.backbone = backbone
        # MMCA中的的降维因子的总乘积随着通道数的翻倍，也跟着翻倍，但为什么变成两个，或者为什么大的放后面，这就无从考究了

        # 2.21新增，在GA模块之前就对resnet+MMCA进行训练，所以这里就添加MLP层
        # self.MLP = nn.Sequential(
        #     nn.Linear(out_channels + genderSize, 1024),
        #     # nn.Linear(out_channels, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     # 3_20改，将结果输出为一个长为230的向量，而不是一个单独的数字
        #     nn.Linear(512, 1)
        #     # nn.Linear(512, 230),
        #     # nn.BatchNorm1d(230),
        #     # nn.ReLU(),
        #     # nn.Linear(230, 1)
        #     # nn.Softmax()
        self.FC0 = nn.Linear(out_channels, 1024)
        self.BN0 = nn.BatchNorm1d(1024)

        self.FC1 = nn.Linear(1024, 512)
        self.BN1 = nn.BatchNorm1d(512)

        self.output = nn.Linear(512, 1)
        # self.output = nn.Linear(512, 240)

    # 前馈函数，需要输入一个图片，以及性别，不仅需要输出feature map，还需要加入MLP输出分类结果
    def forward(self, image):
    # # def forward(self, image):
        # 第一步：用主干网络生成feature_map
        x = self.backbone(image)

        # 第二步：将feature_map降维成texture，这里采用自适应平均池化
        x = F.adaptive_avg_pool2d(x, 1) # N(2048)(H/32)(W/32) -> N(2048)(1)(1)
        # 把后面两个1去除，用torch.squeeze
        x = torch.squeeze(x)
        # 调整x的形状，使dim=1=输出通道的大小
        x = x.view(-1, self.out_channels)

        # output_beforeGA = self.MLP(x)
        # 拆分MLP
        x = F.relu(self.BN0(self.FC0(x)))
        x = F.relu(self.BN1(self.FC1(x)))
        output_beforeGA = self.output(x)

        return output_beforeGA
    # 加入微调函数
    def fine_tune(self, need_fine_tune = True):
        self.train(need_fine_tune)


if __name__ == '__main__':
    x = torch.ones([10, 3, 512, 512], dtype=torch.float32)
    x = x.cuda()
    M = 4
    net = RA_Net(*get_ResNet(), M).cuda()
    net.fine_tune()
    y_hat, y_RA, P = net(x, False)

    print(f"y_hat is :\n{y_hat}\ny_RA is :\n{y_RA}\nP is :\n{P}")
