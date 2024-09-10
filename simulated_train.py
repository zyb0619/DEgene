import argparse
import os
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import configs
from src.models import GcnEncoderNode
from src.train import train_simulated, MSE, evaluate, train_and_val, train_gc, train_syn
from src.utils import *
from utils.io_utils import fix_seed


def main():
    # python3 simulated_train.py --save=true --dataset='simulated' --model='GCN' --multiclass=True
    # python3 simulated_train.py --save=true --dataset='sim1' --model='GCN'

    args = configs.arg_parse()
    fix_seed(args.seed)


    # Load the dataset
    data_path = 'data/sim2.pth'
    data = torch.load(data_path)

    # For pytorch geometric model 
    #model = GCNNet(args.input_dim, args.hidden_dim,
    #       data.num_classes, args.num_gc_layers, args=args)
    input_dims = data.x.shape[-1]
    model = GcnEncoderNode(data.num_features,
                            args.hidden_dim,
                            args.output_dim,
                            data.num_classes,
                            args.num_gc_layers,
                            bn=args.bn,
                            dropout=args.dropout,
                            args=args)
    # train_simulated(data, model, args)
    # test_MSE = MSE(data, model, data.test_mask)
    # print('Test MSE is {:.4f}'.format(test_MSE))
    train_syn(data, model, args)
    _, test_acc = evaluate(data, model, data.test_mask)
    print('Test accuracy is {:.4f}'.format(test_acc))
    
    # Save model 
    model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
    torch.save(model, model_path)


if __name__ == "__main__":
    main()
