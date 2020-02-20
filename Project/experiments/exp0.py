import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import hydra
import yaml
import datetime
import torch.optim as optim
import torch.nn as nn
from fastprogress import master_bar
from src.train_data import train, valid
from src.utils import seed_setting
from src.data.make_dataset import get_train_val
from torchvision import models


def main():
    now = datetime.datetime.now()
    now_date = '{}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    print('{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
    #### パラメータの読み込み ###
    with open('../params/param.yaml', "r+") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    param["date"] = now_date

    seed_setting(param["seed"])
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    trainloader, valloader = get_train_val()

    # model定義
    model = models.alexnet(pretrained=True)
    param['model'] = model.__class__.__name__

    model = model.to(param['device'])



    # loss関数の定義
    criterion = nn.CrossEntropyLoss().to(param["device"])
    # 最適化関数の定義
    if param["optim"].lower() == "sgd":    
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif param["optim"].lower() == "adam":
        optimizer = optim.Adam()

    mb = master_bar(range(param['epoch']))


    # 学習の開始
    train_acc_list = []
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in mb:
        acc, loss = train(trainloader, model, optimizer, criterion, param["device"], parent=mb)
        val_acc, val_loss = valid(valloader, model, criterion, param["device"])
        print('epoch %d, acc: %f, loss: %.4f val_acc: %.4f val_loss: %.4f'
                % (epoch,  acc, loss, val_acc, val_loss))
        train_acc_list.append(acc)
        train_loss_list.append(loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
            

if __name__ == "__main__":
    main()