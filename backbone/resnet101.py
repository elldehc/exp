from typing import Tuple

import torchvision
from torch import nn

import backbone.base


class ResNet101(backbone.base.Base):

    def __init__(self, pretrained: bool):
        super().__init__(pretrained)

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        resnet101 = torchvision.models.resnet101(pretrained=self._pretrained)

        # list(resnet101.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet101.children())
        features = children[:-3]
        num_features_out = 1024

        hidden = children[-3]
        num_hidden_out = 2048

        for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
            for parameter in parameters:
                parameter.requires_grad = False

        features = nn.Sequential(*features)

        return features, hidden, num_features_out, num_hidden_out


# split position: 
# 0: 0
# 1: 3
# 2: 4
# 3: 5
# 4: 6
# 5: 7

def resnet101front(split_pos,pretrained: bool):
    if split_pos==0:
        split_pos=0
    else:
        split_pos=split_pos+2
    resnet101 = torchvision.models.resnet101(pretrained=pretrained)
    children = list(resnet101.children())
    features = children[:split_pos]
    for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
        for parameter in parameters:
            parameter.requires_grad = False

    features = nn.Sequential(*features)
    return features

class ResNet101Back(backbone.base.Base):

    def __init__(self,split_pos,pretrained: bool):
        super().__init__(pretrained)
        assert split_pos>=0 and split_pos<=5
        if split_pos==0:
            self.split_pos=0
        else:
            self.split_pos=split_pos+2

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        resnet101 = torchvision.models.resnet101(pretrained=self._pretrained)

        # list(resnet101.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear
        children = list(resnet101.children())
        features = children[self.split_pos:-3]
        bn_features=children[:-3]
        num_features_out = 1024

        hidden = children[-3]
        num_hidden_out = 2048

        for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
            for parameter in parameters:
                parameter.requires_grad = False

        features = nn.Sequential(*features)

        return features, hidden, num_features_out, num_hidden_out