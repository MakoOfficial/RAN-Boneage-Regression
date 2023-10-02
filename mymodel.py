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
            nn.Linear(512, 240),
            nn.BatchNorm1d(240),
            nn.ReLU(),
            nn.Linear(240, 1)
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


if __name__ == '__main__':
    x = torch.ones([10, 3, 512, 512], dtype=torch.float32)
    x = x.cuda()
    M = 4
    net = RA_Net(*get_ResNet(), M).cuda()
    net.fine_tune()
    y_hat, y_RA, P = net(x, False)

    print(f"y_hat is :\n{y_hat}\ny_RA is :\n{y_RA}\nP is :\n{P}")
