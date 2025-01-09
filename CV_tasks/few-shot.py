import random
from model import Resnet, Attention_Score
import torch
import argparse
from dataset import load_data, CVDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
import numpy as np
import torch.nn.functional as F
import math
from func import mk_mmd_loss

domain_weight=1

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--data_path",type=str,default="./data")
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--class_num', type=int, default=10)

    parser.add_argument("--norm", action="store_true",default=False)
    parser.add_argument("--weight_norm", action="store_true",default=False)
    parser.add_argument("--pretrain_MMD", action="store_true",default=False)
    parser.add_argument("--test_MMD", action="store_true",default=False)
    parser.add_argument('--head_num', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=64)

    parser.add_argument('--shot_num', type=int, default=5)
    parser.add_argument("--model_path", type=str, default="./")

    parser.add_argument('--score', type=str, default="attention") # "attention" or "distance" or "cosine"
    parser.add_argument('--task', type=str, default="digit") # "office" or "digit"

    args = parser.parse_args()
    return args


def expand(x_list, label_list, expand_to_num=64):
    num=x_list.shape[0]
    repeat_num=expand_to_num//num
    x_list=x_list.repeat(repeat_num,1,1,1)
    label_list=label_list.repeat(repeat_num)
    last_num=expand_to_num % num
    for j in range(last_num):
        i=np.random.randint(0,num)
        x_list = torch.cat([x_list, x_list[i:i + 1]], dim=0)
        label_list = torch.cat([label_list, label_list[i:i + 1]], dim=0)
    return x_list,label_list


def few_shot(data_loader, class_num, shot_num,expand_to_num=None):
    x_list=None
    label_list=None

    current_num=[0]*class_num
    dataloader_iterator = iter(data_loader)

    for x, label in dataloader_iterator:

        for i in range(x.shape[0]):
            if current_num[label[i]]==shot_num:
                continue
            current_num[label[i]]+=1
            if x_list is None:
                x_list=x[i:i+1]
                label_list=label[i:i+1]
            else:
                x_list=torch.cat([x_list,x[i:i+1]],dim=0)
                label_list=torch.cat([label_list,label[i:i+1]],dim=0)
    if expand_to_num is not None:
        x_list, label_list = expand(x_list, label_list, expand_to_num)
    return CVDataset(x_list,label_list)




def pre_train(model, attn_model, data_loader, domain_loader, loss_func, optim, device, class_num, train=True, MMD=False):
    w=class_num

    if train:
        model.train()
        attn_model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        attn_model.eval()
        torch.set_grad_enabled(False)
    loss_list = []
    acc_list = []
    pbar = tqdm.tqdm(data_loader, disable=False)

    for x, label in pbar:
        x=x.to(device)
        label = label.to(device)
        y_hat=model(x)
        score=attn_model(y_hat,y_hat)
        num=label.shape[0]
        y=label.unsqueeze(1).repeat(1,num)
        y=(y==y.t()).float()
        if attn_model.score=="distance":
            loss=(score*(y!=0))**2+((3-score)*(y==0))**2
        else:
            loss=loss_func(score,y)
        loss[y>0.5]*=w
        loss=torch.mean(loss)
        loss_list.append(loss.item())

        if train:
            if MMD:
                # y_hat_mean = torch.mean(y_hat, dim=0, keepdim=True)
                # y_hat = y_hat_mean

                dataloader_iterator = iter(domain_loader)
                x_target, _ = next(dataloader_iterator)
                x_target=x_target.to(device)
                y_target = model(x_target)
                # y_hat_target_mean = torch.mean(y_target, dim=0, keepdim=True)
                # y_hat_target = y_hat_target_mean

                # loss_mmd = torch.mean(loss_func(y_hat_target, y_hat))
                loss_mmd=mk_mmd_loss(y_target,y_hat)
                loss += domain_weight * loss_mmd

            optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
            optim.step()

        score[score>0.5]=1
        score[score<=0.5]=0
        if attn_model.score=="distance":
            score=1-score
        acc=torch.mean((score.int()==y.int()).float())
        acc_list.append(acc.item())

    return np.mean(loss_list), np.mean(acc_list)

def iteration(model, attn_model, weight_model, train_loader, test_loader, loss_func, optim, device, class_num, hidden_dim, train=True, MMD=False, batch_size=64):
    w=class_num

    if train:
        model.train()
        attn_model.train()
        weight_model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        attn_model.eval()
        weight_model.train()
        torch.set_grad_enabled(False)

    loss_list = []
    acc_list = []
    if train:
        data_loader=train_loader
        domain_loader=test_loader
    else:
        data_loader=test_loader

    pbar = tqdm.tqdm(data_loader, disable=False)
    for x, label_ in pbar:

        # generate template
        template=torch.zeros([class_num,hidden_dim]).to(device)
        template_weights=torch.zeros([class_num,1]).to(device)
        for j in range(2):
            dataloader_iterator = iter(train_loader)
            x_train, label = next(dataloader_iterator)
            if x_train.shape[0]<batch_size:
                x_train, label = expand(x_train, label, batch_size)
            x_train = x_train.to(device)
            label = label.to(device)
            y_train = model(x_train)
            score = attn_model(y_train, y_train)
            score = score.unsqueeze(0)
            score = score.unsqueeze(0)
            weight = weight_model(score)
            weight = weight.squeeze()
            weight = F.sigmoid(weight)
            num = y_train.shape[0]
            for i in range(num):
                template[label[i]] += y_train[i] * weight[i]
                template_weights[label[i]] += weight[i]
        template=template/template_weights

        x=x.to(device)
        label = label_.to(device)
        y_hat=model(x)
        score=attn_model(y_hat,template)
        if attn_model.score=="distance":
            output=torch.argmin(score, dim=-1)
        else:
            output=torch.argmax(score, dim=-1)
        acc=torch.mean((output==label).float())
        num=label.shape[0]
        y=torch.zeros([num,class_num]).to(device)
        for i in range(num):
            y[i,label[i]]=1

        if attn_model.score == "distance":
            loss = (score * (y != 0)) ** 2 + ((3 - score) * (y == 0)) ** 2
        else:
            loss = loss_func(score, y)
        loss[y>0.5]*=w
        loss=torch.mean(loss)
        loss_list.append(loss.item())

        if train:
            if MMD:
                # y_hat_mean = torch.mean(y_hat, dim=0, keepdim=True)
                # y_hat = y_hat_mean

                dataloader_iterator = iter(domain_loader)
                x_target, _ = next(dataloader_iterator)
                x_target=x_target.to(device)
                y_target = model(x_target)
                # y_hat_target_mean = torch.mean(y_target, dim=0, keepdim=True)
                # y_hat_target = y_hat_target_mean

                # loss_mmd = torch.mean(loss_func(y_hat_target, y_hat))
                loss_mmd=mk_mmd_loss(y_target,y_hat)
                loss += domain_weight * loss_mmd

            optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
            optim.step()

        acc_list.append(acc.item())

    return np.mean(loss_list), np.mean(acc_list)

def main():
    args=get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    class_num = args.class_num
    pretrain = True
    if args.task == "office":
        model = Resnet(output_dims=args.hidden_dim, channel=3, pretrained=pretrain, norm=args.norm)
    else:
        model = Resnet(output_dims=args.hidden_dim, channel=1, pretrained=pretrain, norm=args.norm)
    model = model.to(device)
    weight_model = Resnet(output_dims=args.batch_size, channel=1, pretrained=pretrain, norm=args.weight_norm)
    weight_model = weight_model.to(device)
    attn_model = Attention_Score(args.hidden_dim, args.hidden_dim, method=args.score)
    attn_model = attn_model.to(device)

    model.load_state_dict(torch.load(args.model_path+args.task+".pth"))
    weight_model.load_state_dict(torch.load(args.model_path+args.task+"_weight.pth"))
    attn_model.load_state_dict(torch.load(args.model_path+args.task+"_attention.pth"))


    parameters = set(model.parameters()) | set(attn_model.parameters()) | set(weight_model.parameters())

    total_params = sum(p.numel() for p in parameters if p.requires_grad)
    print('total parameters:', total_params)
    optim = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0.01)

    domain_data, test_data = load_data(task=args.task)

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    domain_loader = DataLoader(domain_data, batch_size=args.batch_size, shuffle=True)

    domain_loader1=domain_loader
    domain_loader2=test_loader

    train_data = few_shot(test_loader, class_num=args.class_num, shot_num=args.shot_num)
    # train_data = few_shot(test_loader, task=args.task, class_num=args.class_num, shot_num=args.shot_num, expand_to_num=args.batch_size)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # loss_func = nn.MSELoss(reduction="none")
    loss_func = nn.BCELoss(reduction="none")

    best_acc=0
    best_loss=1000
    acc_epoch=0
    loss_epoch=0
    j=0

    loss, acc = iteration(model, attn_model, weight_model, train_loader, test_loader, loss_func, optim, device,
            class_num, args.hidden_dim, train=False, MMD=False, batch_size=args.batch_size)
    log = "Initial Loss {:06f}, Initial Acc {:06f} ".format(loss,acc)
    print(log)

    while True:
        j+=1

        pre_train(model, attn_model, train_loader, domain_loader1, loss_func, optim, device,
                  class_num, train=True,  MMD=args.pretrain_MMD)

        loss, acc = iteration(model, attn_model, weight_model, train_loader, domain_loader2, loss_func, optim, device,
                 class_num, args.hidden_dim, train=True, MMD=args.test_MMD, batch_size=args.batch_size)
        log = "Epoch {} | Train Loss {:06f},  Train Acc {:06f} | ".format(j, loss, acc)
        print(log)
        with open(args.task+"_finetune.txt", 'a') as file:
            file.write(log)

        loss, acc = iteration(model, attn_model, weight_model, train_loader, test_loader, loss_func, optim, device,
                  class_num, args.hidden_dim, train=False, MMD=False, batch_size=args.batch_size)
        log = "Test Loss {:06f}, Test Acc {:06f} ".format(loss,acc)
        print(log)
        with open(args.task+"_finetune.txt", 'a') as file:
            file.write(log+"\n")

        if acc >= best_acc or loss <= best_loss:
            torch.save(model.state_dict(), args.task + "_finetune.pth")
            torch.save(weight_model.state_dict(), args.task + "_weight_finetune.pth")
            torch.save(attn_model.state_dict(), args.task + "_attention_finetune.pth")
        if acc >= best_acc:
            best_acc = acc
            acc_epoch = 0
        else:
            acc_epoch += 1
        if loss < best_loss:
            best_loss = loss
            loss_epoch = 0
        else:
            loss_epoch += 1
        if acc_epoch >= args.epoch and loss_epoch >= args.epoch:
            break
        print("Acc Epoch {:}, Loss Epcoh {:}".format(acc_epoch, loss_epoch))



if __name__ == '__main__':
    main()