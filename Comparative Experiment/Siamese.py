from model import Resnet
import torch
import argparse
from dataset import load_data, load_zero_shot
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
import numpy as np
import copy
import torch.nn.functional as F
import math
from func import mk_mmd_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--data_path",type=str,default="../data")
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument("--test_list", type=int, nargs='+', default=[0])
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--class_num', type=int, default=6) # action:6, people:8
    parser.add_argument('--task', type=str, default="action") # "action" or "people"

    parser.add_argument("--norm", action="store_true",default=False)
    parser.add_argument("--cross_class", action="store_true",default=False)


    args = parser.parse_args()
    return args

def iteration(model, data_loader, optim, device, task, class_num, hidden_dim,train=True):
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)
    loss_list = []
    acc_list = []
    pbar = tqdm.tqdm(data_loader, disable=False)


    if train:
        for x, action, people in pbar:
            x = x.to(device)
            if task == "action":
                label = action.to(device)
            elif task == "people":
                label = people.to(device)
            else:
                print("ERROR")
                exit(-1)
            y_hat,_=model(x)
            dis=torch.cdist(y_hat, y_hat, p=2)
            num = label.shape[0]
            y = label.unsqueeze(1).repeat(1, num)
            y = (y == y.t()).float()
            output=(dis<0.5).float()
            loss=(dis*(y!=0))**2+((3-dis)*(y==0))**2
            loss[y > 0.5] = loss[y > 0.5]* class_num
            loss=torch.mean(loss)
            model.zero_grad()
            loss.backward()
            optim.step()
            acc=torch.mean((output==y).float())
            loss_list.append(loss.item())
            acc_list.append(acc.item())

    else:
        template = torch.zeros([class_num, hidden_dim]).to(device)
        num = 0
        for x, action, people in data_loader:
            x = x.to(device)
            if task == "action":
                label = action.to(device)
            elif task == "people":
                label = people.to(device)
            else:
                print("ERROR")
                exit(-1)
            y,_ = model(x)
            for i in range(x.shape[0]):
                if torch.sum(template[label[i]]) == 0:
                    template[label[i]] = y[i]
                    num += 1
                if num == class_num:
                    break
            if num == class_num:
                break


        for x, action, people in pbar:
            x = x.to(device)
            if task == "action":
                label = action.to(device)
            elif task == "people":
                label = people.to(device)
            else:
                print("ERROR")
                exit(-1)
            y_hat,_ = model(x)
            dis = torch.cdist(y_hat, template, p=2)
            output = torch.argmin(dis, dim=-1)
            acc = torch.mean((output == label).float())
            num = label.shape[0]
            y = torch.zeros([num, class_num]).to(device)
            for i in range(num):
                y[i, label[i]] = 1

            loss=(dis*(y!=0))**2+((3-dis)*(y==0))**2
            loss[y > 0.5] *= class_num
            loss=torch.mean(loss)
            loss_list.append(loss.item())
            acc_list.append(acc.item())

    return np.mean(loss_list), np.mean(acc_list)


def main():
    args=get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    class_num = args.class_num
    model = Resnet(output_dims=64, pretrained=True, norm=args.norm)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total parameters:', total_params)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)


    if not args.cross_class:
        if args.task=="action":
            train_data, test_data = load_zero_shot(test_people_list=args.test_list, data_path=args.data_path)
        elif args.task=="people":
            train_data, test_data = load_zero_shot(test_action_list=args.test_list, data_path=args.data_path)
        else:
            print("ERROR")
            exit(-1)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    else:
        if args.task == "action":
            train_data, test_data1 = load_zero_shot(test_people_list=args.test_list, data_path=args.data_path)
        elif args.task == "people":
            train_data, test_data1 = load_zero_shot(test_action_list=args.test_list, data_path=args.data_path)
        else:
            print("ERROR")
            exit(-1)
        train_data, test_data2 = train_test_split(train_data, test_size=0.1)
        test_data = ConcatDataset([test_data1, test_data2])

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)



    best_acc=0
    best_loss=1000
    acc_epoch=0
    loss_epoch=0
    j=0
    while True:
        j+=1
        loss, acc = iteration(model, train_loader, optim, device, args.task,class_num, 64, train=True)
        log = "Epoch {} | Train Loss {:06f},  Train Acc {:06f} | ".format(j, loss, acc)
        print(log)
        with open(args.task + ".txt", 'a') as file:
            file.write(log)
        loss, acc = iteration(model, test_loader, optim, device, args.task,class_num, 64, train=False)
        log = "Test Loss {:06f}, Test Acc {:06f} ".format(loss, acc)
        print(log)
        with open(args.task + ".txt", 'a') as file:
            file.write(log + "\n")

        if acc >= best_acc or loss <= best_loss:
            torch.save(model.state_dict(), args.task + ".pth")
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