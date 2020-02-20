import torch
import hydra
import yaml


def main():
    #### パラメータの設定 ###
    with open('../params/param.yaml', "r+") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    