from model import Resnet, Attention_Score, DANN
import torch
import argparse
from dataset import load_data, load_zero_shot
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
import numpy as np
import torch.nn.functional as F
import math
import copy
from func import mk_mmd_loss

domain_weight=1

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--data_path",type=str,default="./data")
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument("--test_list", type=int, nargs='+', default=[0])
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--class_num', type=int, default=6) # action:6, people:8
    parser.add_argument('--task', type=str, default="action") # "action" or "people"

    parser.add_argument("--norm", action="store_true",default=False)
    parser.add_argument("--weight_norm", action="store_true",default=False)
    parser.add_argument("--adversarial", action="store_true",default=False)
    parser.add_argument("--MMD", action="store_true",default=False)
    parser.add_argument('--head_num', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=64)

    parser.add_argument('--template', type=str, default="WeightNet") # "WeightNet" or "Average" or "Random"
    parser.add_argument('--score', type=str, default="attention") # "distance" or "cosine"

    args = parser.parse_args()
    return args

def pre_train(model, attn_model, dann, data_loader, domain_loader, loss_func, loss_cls, optim, device, task, class_num, train=True, adversarial=False, alpha=1.0, MMD=False):
    w=class_num

    if train:
        model.train()
        attn_model.train()
        dann.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        attn_model.eval()
        dann.eval()
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


        if train:
            if MMD:
                # y_hat_mean = torch.mean(y_hat, dim=0, keepdim=True)
                # y_hat = y_hat_mean

                dataloader_iterator = iter(domain_loader)
                x_target, _, _ = next(dataloader_iterator)
                x_target=x_target.to(device)
                y_target = model(x_target)
                # y_hat_target_mean = torch.mean(y_target, dim=0, keepdim=True)
                # y_hat_target = y_hat_target_mean

                # loss_mmd = torch.mean(loss_func(y_hat_target, y_hat))
                loss_mmd=mk_mmd_loss(y_target,y_hat)
                loss += domain_weight * loss_mmd

            if adversarial:
                truth = torch.ones_like(label).to(device)
                truth_hat = dann(x, alpha=alpha)
                loss_truth = loss_cls(truth_hat, truth)

                dataloader_iterator = iter(domain_loader)
                x_false, label_false, _ = next(dataloader_iterator)
                x_false = x_false.to(device)
                false = torch.zeros_like(label_false).to(device)
                false_hat = dann(x_false, alpha=alpha)
                loss_false = loss_cls(false_hat, false)

                loss += (0.5 * loss_truth + 0.5 * loss_false) * domain_weight

            model.zero_grad()
            attn_model.zero_grad()
            dann.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
            optim.step()

        score[score>0.5]=1
        score[score<=0.5]=0
        if attn_model.score=="distance":
            score=1-score
        acc=torch.mean((score.int()==y.int()).float())

        loss_list.append(loss.item())
        acc_list.append(acc.item())

    return np.mean(loss_list), np.mean(acc_list)

def iteration(model, attn_model, weight_model, dann, train_loader, test_loader, loss_func, loss_cls, optim, device, task, class_num, hidden_dim, train=True, adversarial=False, alpha=1.0, MMD=False, template_method="WeightNet"):
    w=class_num

    loss_list = []
    acc_list = []
    if train:
        data_loader=train_loader
        domain_loader=test_loader
    else:
        data_loader=test_loader

    model.eval()
    attn_model.eval()
    weight_model.train()
    dann.eval()
    torch.set_grad_enabled(False)

    if template_method=="WeightNet":
        # flag=False
        # if flag:
        if MMD:
            template=torch.zeros([class_num,hidden_dim]).to(device)
            num=0
            for x,action,people in train_loader:
                x = x.to(device)
                if task == "action":
                    label = action.to(device)
                elif task == "people":
                    label = people.to(device)
                else:
                    print("ERROR")
                    exit(-1)
                y = model(x)
                for i in range(x.shape[0]):
                    if torch.sum(template[label[i]])==0:
                        template[label[i]]=y[i]
                        num+=1
                    if num==class_num:
                        break
                if num==class_num:
                    break
        else:
            template = torch.zeros([class_num, hidden_dim]).to(device)
            template_weights = torch.zeros([class_num, 1]).to(device)
            dataloader_iterator = iter(train_loader)
            for j in range(2):
                x_train, action_train, people_train = next(dataloader_iterator)
                x_train = x_train.to(device)
                if task == "action":
                    label = action_train.to(device)
                elif task == "people":
                    label = people_train.to(device)
                else:
                    print("ERROR")
                    exit(-1)
                y_train = model(x_train)
                score = attn_model(y_train, y_train)
                score = score.unsqueeze(0)
                score = score.unsqueeze(0)
                weight = weight_model(score)
                weight = weight.squeeze()
                weight = F.sigmoid(weight)
                num = y_train.shape[0]
                for i in range(num):
                    if weight[i]>template_weights[label[i]]:
                        template_weights[label[i]]=weight[i]
                        template[label[i]]=y_train[i]

        if not train:
            # find the most similar sample as new template
            new_template = copy.deepcopy(template).to(device)
            template_sim = torch.zeros([class_num]).to(device)
            template_label = torch.zeros([class_num]).to(device) - 1
            j=0
            for x, action, people in data_loader:
                j+=1
                x = x.to(device)
                if task == "action":
                    label = action.to(device)
                elif task == "people":
                    label = people.to(device)
                else:
                    print("ERROR")
                    exit(-1)
                y = model(x)
                score = attn_model(y, template)
                output = torch.argmax(score, dim=-1).int()
                for i in range(y.shape[0]):
                    if score[i, output[i]] > template_sim[output[i]]:
                        if train and label[i] != output[i]:
                            continue
                        template_sim[output[i]] = score[i, output[i]]
                        new_template[output[i]] = y[i]
                        template_label[output[i]] = label[i]
                if j>=3:
                    break
            # print(template_label)
            template = new_template
    elif template_method == "random":
        template = torch.zeros([class_num, hidden_dim]).to(device)
        num = 0
        for x_train, action_train, people_train in train_loader:
            x_train = x_train.to(device)
            if task == "action":
                label = action_train.to(device)
            elif task == "people":
                label = people_train.to(device)
            else:
                print("ERROR")
                exit(-1)
            y_train = model(x_train)
            for i in range(x_train.shape[0]):
                if torch.sum(template[label[i]]) == 0:
                    template[label[i]] = y_train[i]
                    num += 1
                if num == class_num:
                    break
            if num == class_num:
                break
    elif template_method == "average":
        template = torch.zeros([class_num, hidden_dim]).to(device)
        template_weights = torch.zeros([class_num, 1]).to(device)
        dataloader_iterator = iter(train_loader)
        for x_train, action_train, people_train in dataloader_iterator:
            x_train = x_train.to(device)
            if task == "action":
                label = action_train.to(device)
            elif task == "people":
                label = people_train.to(device)
            else:
                print("ERROR")
                exit(-1)
            y_train = model(x_train)
            num = y_train.shape[0]
            for i in range(num):
                template[label[i]] += y_train[i]
                template_weights[label[i]] += 1
        template = template / template_weights
    else:
        print("ERROR")
        exit(-1)

    if train:
        model.train()
        attn_model.train()
        weight_model.train()
        dann.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        attn_model.eval()
        weight_model.train()
        dann.eval()
        torch.set_grad_enabled(False)
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

        if train:
            if MMD:
                # y_hat_mean = torch.mean(y_hat, dim=0, keepdim=True)
                # y_hat = y_hat_mean

                dataloader_iterator = iter(domain_loader)
                x_target, _, _ = next(dataloader_iterator)
                x_target=x_target.to(device)
                y_target = model(x_target)
                # y_hat_target_mean = torch.mean(y_target, dim=0, keepdim=True)
                # y_hat_target = y_hat_target_mean

                # loss_mmd = torch.mean(loss_func(y_hat_target, y_hat))
                loss_mmd=mk_mmd_loss(y_target,y_hat)
                loss += domain_weight * loss_mmd

            if adversarial:
                truth = torch.ones_like(label).to(device)
                truth_hat = dann(x, alpha=alpha)
                loss_truth = loss_cls(truth_hat, truth)

                dataloader_iterator = iter(domain_loader)
                x_false, label_false, _ = next(dataloader_iterator)
                x_false = x_false.to(device)
                false = torch.zeros_like(label_false).to(device)
                false_hat = dann(x_false, alpha=alpha)
                loss_false = loss_cls(false_hat, false)

                loss += (0.5 * loss_truth + 0.5 * loss_false) * domain_weight

            model.zero_grad()
            attn_model.zero_grad()
            weight_model.zero_grad()
            dann.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 3.0)  # 用于裁剪梯度，防止梯度爆炸
            optim.step()

        loss_list.append(loss.item())
        acc_list.append(acc.item())

    return np.mean(loss_list), np.mean(acc_list)

def main():
    args=get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')
    class_num = args.class_num
    model = Resnet(output_dims=args.hidden_dim, channel=2, pretrained=True, norm=args.norm)
    model = model.to(device)
    weight_model = Resnet(output_dims=args.batch_size,channel=1,pretrained=True, norm=args.weight_norm)
    weight_model = weight_model.to(device)
    attn_model = Attention_Score(args.hidden_dim,args.hidden_dim,method=args.score)
    attn_model = attn_model.to(device)
    dann = DANN(model, args.hidden_dim)
    dann = dann.to(device)

    parameters = set(model.parameters()) | set(attn_model.parameters()) | set(weight_model.parameters()) | set(dann.parameters())

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
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    loss_func = nn.MSELoss(reduction="none")
    loss_cls = nn.CrossEntropyLoss()

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
        pre_train(model, attn_model, dann, train_loader, test_loader, loss_func, loss_cls, optim, device, args.task,
                  class_num, train=True, adversarial=args.adversarial, alpha=alpha, MMD=args.MMD)

        loss, acc = iteration(model, attn_model, weight_model, dann, train_loader, test_loader, loss_func, loss_cls, optim, device,
                  args.task, class_num, args.hidden_dim, train=True, adversarial=args.adversarial, alpha=alpha, MMD=args.MMD,template_method=args.template)
        log = "Epoch {} | Train Loss {:06f},  Train Acc {:06f} | ".format(j, loss, acc)
        print(log)
        with open(args.task+".txt", 'a') as file:
            file.write(log)

        loss, acc = iteration(model, attn_model, weight_model, dann, train_loader, test_loader, loss_func, loss_cls, optim, device,
                  args.task, class_num, args.hidden_dim, train=False, adversarial=False, alpha=alpha, MMD=False,template_method=args.template)
        log = "Test Loss {:06f}, Test Acc {:06f} ".format(loss,acc)
        print(log)
        with open(args.task+".txt", 'a') as file:
            file.write(log+"\n")

        if acc >= best_acc or loss <= best_loss:
            torch.save(model.state_dict(), args.task + ".pth")
            torch.save(weight_model.state_dict(), args.task + "_weight.pth")
            torch.save(attn_model.state_dict(), args.task + "_attention.pth")
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