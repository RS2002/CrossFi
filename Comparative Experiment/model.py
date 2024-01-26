import torch.nn as nn
import torchvision.models as models
import torch
from torch.autograd import Function


class Resnet(nn.Module):
    def __init__(self, output_dims, channel=2, pretrained=True, norm=False):
        super().__init__()
        self.model=models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 64)
        self.linear = nn.Linear(64, output_dims)
        self.norm=norm

    def forward(self,x):
        # x=x[:,0:1,...]
        if self.norm:
            mean=torch.mean(x,dim=-1,keepdim=True)
            std=torch.std(x,dim=-1,keepdim=True)
            y=(x-mean)/std
        else:
            y=x
        y=self.model(y)
        return self.linear(y), y

class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN(nn.Module):
    def __init__(self,model,hidden_dim=64):
        super().__init__()
        self.model=model
        self.linear=nn.Linear(hidden_dim,2)
        self.GRL = GRL()

    def forward(self,x,alpha=1):
        _,y=self.model(x)
        y = self.GRL.apply(y,alpha)
        y=self.linear(y)
        return y