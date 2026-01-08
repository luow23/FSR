
from torchvision.models import vgg16_bn, vgg19_bn, resnet34, resnet18, mobilenet_v2
from torch import nn
import torch

class PretrainedFeature(nn.Module):
    def __init__(self, model_name):
        super(PretrainedFeature, self).__init__()
        if "vgg" in model_name:
            model = eval(model_name)(pretrained=True)
            # print(model.features)
            self.features = model.features
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif "resnet" in model_name:
            model = eval(model_name)(pretrained=True)
            self.features = nn.Sequential(*list(model.children)[:-2])
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        else:
            raise NotImplementedError

    def forward(self, x):
        feature_vector = self.pool(self.features(x))
        return feature_vector.squeeze(-1).squeeze(-1).detach()

if __name__ == '__main__':
    model = resnet18(pretrained=False)
    print(list(model.children()))
