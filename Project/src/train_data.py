import torch 
from torchvision import models
from fastprogress import progress_bar
import torch.nn as nn

# 訓練データの学習行う関数
def train(dataloader, model, optimizer, criterion, device):
    # modelのモード変更
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for step, data in enumerate(progress_bar(dataloader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = torch.tensor(inputs, dtype=torch.float).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        
        # 正解率の計算
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels.data).sum()
        total += labels.size(0)

        loss.backward()
        optimizer.step()
        # print statistics
    train_acc = float(correct) / total
    train_loss = running_loss / len(dataloader)
    return train_acc, train_loss


def valid(dataloader, model, criterion, device):
    # モデルのテスト
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = torch.tensor(inputs, dtype=torch.float).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            outputs = model(inputs)
            # lossの計算
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # 正解率の計算
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = running_loss / len(dataloader)
    val_acc = float(correct) / total
    return val_acc, val_loss


# ネットワークの初期化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)