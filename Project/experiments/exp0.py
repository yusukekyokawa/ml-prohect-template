import torch
import hydra
import yaml
imprt 

def main():

    now = datetime.datetime.now()
    now_date = '{}-{:0>2d}-{:0>2d}_{:0>2d}-{:0>2d}-{:0>2d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    print('{}-{}-{} {}:{}:{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second))
    #### パラメータの設定 ###
    with open('../params/param.yaml', "r+") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)


    