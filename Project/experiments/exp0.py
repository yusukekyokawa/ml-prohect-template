import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import yaml
import datetime
import torch.optim as optim
import torch.nn as nn
from src.train_data import train, valid
from src.utils import seed_setting
from src.data.make_dataset import get_train_val
from torchvision import models
from fastprogress import master_bar, progress_bar


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

    if param["device"] == "cpu":
        device = torch.device("cpu")
    elif param["device"] == "gpu":
        device = torch.device("cuda")
    
    trainloader, valloader = get_train_val()

    # model定義
    model = models.alexnet(pretrained=True)
    param['model'] = model.__class__.__name__
    
    
    model = model.to(device)



    # loss関数の定義
    criterion = nn.CrossEntropyLoss().to(device)
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
        acc, loss = train(trainloader, model, optimizer, criterion, device, parent=mb)
        val_acc, val_loss = valid(valloader, model, criterion, device)
        print('======================== epoch {} ========================'.format(epoch+1))
        # print('lr              : {:.5f}'.format(scheduler.get_lr()[0]))
        print('loss            : train={:.5f}  , test={:.5f}'.format(loss, val_loss))
        print('acc : train={:.3%}  , test={:.3%}'.format(acc, val_acc))

        train_acc_list.append(acc)
        train_loss_list.append(loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
            

if __name__ == "__main__":
    main()