import argparse
import random
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

import configs
from utils.io_utils import fix_seed
from src.data import prepare_data
from src.explainers import GraphSVX
from src.train import MSE, evaluate, test


def main():

    # python3 simulated_explain.py --dataset='simulated' --model='GCN' --info=True  --multiclass=True
    # python3 simulated_explain.py --dataset='sim1' --model='GCN' --info=True

    args = configs.arg_parse()
    fix_seed(args.seed)

    # Load the dataset
    data_path = 'data/sim2.pth'
    data = torch.load(data_path)

    # Load the model
    model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
    model = torch.load(model_path)
    
    # Evaluate the model 
    # test_MSE = MSE(data, model, data.test_mask)
    # print('Test MSE is {:.4f}'.format(test_MSE))
    test_acc = test(data, model, data.test_mask)
    print('Test accuracy is {:.4f}'.format(test_acc))
    
    # Explain it with GraphSVX
    explainer = GraphSVX(data, model, args.gpu)

    sorted_counts = {}

    for i in range(1,7):
        group_attr = f"group{i}"
        data_group = getattr(data, group_attr)
        # Distinguish graph classfication from node classification
        explanations_group,genelist,contribution_list = explainer.explain(
                                            # 设置每个类群中随机采样出的cell_index
                                            random.sample(data_group, 30),
                                            # args.hops,
                                            2,
                                            # args.num_samples,
                                            # 设置每个cell采样的掩码数量
                                            5000,
                                            args.info,
                                            args.multiclass,
                                            args.fullempty,
                                            # args.S,
                                            2,
                                            args.hv,
                                            args.feat,
                                            args.coal,
                                            # 'All',
                                            args.g,
                                            args.regu,
                                            # 1,
                                            True)
        print('Sum explanations: ', [np.sum(explanation) for explanation in explanations_group])
        print('Base value: ', explainer.base_values)
        df = pd.DataFrame({'gene': genelist, 'score': contribution_list})
        summarized_df = df.groupby('gene', as_index=False).sum()
        sorted_df = summarized_df.sort_values(by='score', ascending=False)
        sorted_counts['group{}'.format(i)] = sorted_df

    for group, gene in sorted_counts.items():
        print( group, 'marker gene: ')
        print(gene.head(5))
        gene.to_csv(group+".csv")


if __name__ == "__main__":
    main()
