import os
import torch
import argparse
from src.model import cfgs
from utils.search_space import PoolSearchSpace
from utils.env import set_seed
from src.model import _vgg

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = "vgg19", type = str, choices = ["vgg16", "vgg19"])
    parser.add_argument("--model_config", default = "E", type = str, choices = ["D", "E"])
    parser.add_argument("--num_configs", default = 100, type = int)
    parser.add_argument("--seed", default = 0, type = int)
    
    args = parser.parse_args()

    set_seed(args.seed)

    model_config = cfgs["E"]
    pool_search_space = PoolSearchSpace("vgg19", model_config, num_configs = 100)
    pool_search_space.create_search_space()

    model = _vgg(pool_search_space.search_space["50"], True, None, True)
    print(model)
