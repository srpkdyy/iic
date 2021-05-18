import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchvision import models

class IICNet(models.ResNet):
    def __init__(self, in_channels=3, n_classes=2, n_heads=1, pretrained=False, semisup=False, **kwargs):
        block = models.resnet.BasicBlock
        super(IICNet, self).__init__(
            block,
            [2, 2, 2, 2],
            **kwargs)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            state_dict = models.utils.load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            )
            self.load_state_dict(state_dict)

        self.trunk = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool
        )
        
        self.n_heads = n_heads
        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(512 * block.expansion, n_classes), nn.Softmax(dim=1)) for _ in range(self.n_heads)
        ])
        self.fc_overclustering = nn.ModuleList([nn.Sequential(
            nn.Linear(512 * block.expansion, n_classes * 5), nn.Softmax(dim=1)) for _ in range(self.n_heads)
        ])

        if semisup:
            self.head = nn.Sequential(
                nn.Linear(512 * block.expansion, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, n_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.normal_(m.bias)

        


    def forward(self, x, semisup=False):
        x = self.trunk(x)
        x = torch.flatten(x, 1)

        if semisup:
            return self.head(x)

        y = []
        y_overclustering = []
        for i in range(self.n_heads):
            y.append(self.fc[i](x))
            y_overclustering.append(self.fc_overclustering[i](x))

        return y, y_overclustering

    
    def init_fc(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.normal_(m.bias)

