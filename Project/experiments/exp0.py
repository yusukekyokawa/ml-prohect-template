import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import hydra
import yaml
import datetime

from src.train_data import train, valid


def main():

    now = datetime.datetime.now()
    now_date = '{}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    print('{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
    #### パラメータの設定 ###
    with open('../params/param.yaml', "r+") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)


    