import torch.nn as nn
import torchvision.models as models
import torch
from torch.autograd import Function

class Resnet(nn.Module):
    def __init__(self, output_dims=64, channel=2, pretrained=True, norm=False):
        super().__init__()
        self.model=models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dims)
        self.norm=norm

    def forward(self,x):
        if self.norm:
            mean=torch.mean(x,dim=-1,keepdim=True)
            std=torch.std(x,dim=-1,keepdim=True)
            y=(x-mean)/std
        else:
            y=x
        return self.model(y)

class Attention_Score(nn.Module):
    def __init__(self, input_dims=64,hidden_dims=64,head=1):
        super().__init__()
        if hidden_dims%head!=0:
            print("ERROR")
            exit(-1)
        self.q_linear=nn.Linear(input_dims,hidden_dims)
        self.k_linear=nn.Linear(input_dims,hidden_dims)
        self.sigmoid=nn.Sigmoid()
        self.head=head
        self.num=hidden_dims//head
        self.input_dims=input_dims


    def forward(self,q,k):
        query=self.q_linear(q)
        key=self.k_linear(k)

        query=query.view(-1,self.head,self.num).transpose(0, 1)
        key=key.view(-1,self.head,self.num).transpose(0, 1)

        attn_matrix=torch.bmm(query,key.transpose(1, 2))
        attn_matrix=torch.sum(attn_matrix,dim=0)


        return self.sigmoid(attn_matrix)


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
        y=self.model(x)
        y = self.GRL.apply(y,alpha)
        y=self.linear(y)
        return y

# test
if __name__ == '__main__':
  q=torch.rand([15,64])
  k=torch.rand([6,64])
  Attn=Attention_Score(head=2)
  s=Attn(q,k)
  print(s.shape)
