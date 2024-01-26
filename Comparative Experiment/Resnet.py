from model import Resnet,DANN
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
    parser.add_argument("--DANN", action="store_true",default=False)
    parser.add_argument("--MMD", action="store_true",default=False)
    parser.add_argument("--MK_MMD", action="store_true",default=False)



    args = parser.parse_args()
    return args

def iteration(model, data_loader, domain_loader,loss_func, optim, device, task, dann, train=True, DANN=False, alpha=1.0, MMD=False, MK_MMD=False):
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    loss_list = []
    acc_list = []
    pbar = tqdm.tqdm(data_loader, disable=False)
    for x, action, people in pbar:
        x=x.to(device)
        if task == "action":
            label = action.to(device)
        elif task == "people":
            label = people.to(device)
        else:
            print("ERROR")
            exit(-1)
        y,y_hidden=model(x)
        loss=loss_func(y,label)
        output=torch.argmax(y, dim=-1)
        acc=torch.mean((output==label).float())

        if train:
            if DANN:
                loss_cls=nn.CrossEntropyLoss()
                truth = torch.ones_like(label).to(device)
                truth_hat = dann(x, alpha=alpha)
                loss_truth = loss_cls(truth_hat, truth)

                dataloader_iterator = iter(domain_loader)
                x_false, label_false, _ = next(dataloader_iterator)
                x_false=x_false.to(device)
                false = torch.zeros_like(label_false).to(device)
                false_hat = dann(x_false, alpha=alpha)
                loss_false = loss_cls(false_hat, false)

                loss += (0.5 * loss_truth + 0.5 * loss_false)

            if MMD or MK_MMD:
                dataloader_iterator = iter(domain_loader)
                x_target, _, _ = next(dataloader_iterator)
                x_target=x_target.to(device)
                _, y_target = model(x_target)
                if MMD:
                    loss_mmd=mk_mmd_loss(y_target,y_hidden,kernel_types=['gaussian'],kernel_params=[2.0])
                else:
                    loss_mmd = mk_mmd_loss(y_target, y_hidden)
                loss += loss_mmd


            model.zero_grad()
            dann.zero_grad()
            loss.backward()
            optim.step()

        loss_list.append(loss.item())
        acc_list.append(acc.item())

    return np.mean(loss_list), np.mean(acc_list)


def main():
    args=get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    class_num = args.class_num
    model = Resnet(output_dims=class_num, pretrained=True, norm=args.norm)
    model = model.to(device)

    dann = DANN(model, 64)
    dann = dann.to(device)

    parameters = set(model.parameters()) | set(dann.parameters())

    total_params = sum(p.numel() for p in parameters if p.requires_grad)
    print('total parameters:', total_params)
    optim = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0.01)

    # train_data, test_data = load_data(args.data_path, train_prop=0.9)
    if args.task=="action":
        train_data, test_data = load_zero_shot(test_people_list=args.test_list, data_path=args.data_path)
    elif args.task=="people":
        train_data, test_data = load_zero_shot(test_action_list=args.test_list, data_path=args.data_path)
    else:
        print("ERROR")
        exit(-1)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    loss_func = nn.CrossEntropyLoss()

    best_acc=0
    best_loss=1000
    acc_epoch=0
    loss_epoch=0
    j=0
    while True:
        j+=1

        num = 100
        if j > num:
            alpha = 1.0
        else:
            alpha = 2.0 / (1.0 + math.exp(-10 * j / num)) - 1

        loss, acc = iteration(model, train_loader, test_loader, loss_func, optim, device, args.task, dann, train=True, DANN=args.DANN, alpha=alpha, MMD=args.MMD, MK_MMD=args.MK_MMD)
        log = "Epoch {} | Train Loss {:06f},  Train Acc {:06f} | ".format(j, loss, acc)
        print(log)
        with open(args.task+".txt", 'a') as file:
            file.write(log)
        loss, acc = iteration(model, test_loader, test_loader, loss_func, optim, device, args.task, dann, train=False)
        log = "Test Loss {:06f}, Test Acc {:06f} ".format(loss,acc)
        print(log)
        with open(args.task+".txt", 'a') as file:
            file.write(log+"\n")

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