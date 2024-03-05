import argparse
import numpy as np
import os
from tqdm import tqdm
import time
import pickle as pkl
import matplotlib.pyplot as plt
import json

import torch
import copy
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from utils.utils import build_spanning_tree_edge, find_higher_order_neighbors, add_self_loops
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

def BFS(edge_index):
    stack = [(0, 0, -1)]
    visited = {}
    while len(stack):
        cur_id, cur_depth, parent_id = stack.pop(0)
        #print(cur_id)
        if cur_id not in visited.keys():
            visited[cur_id] = {
                'id': cur_id, 
                'depth': cur_depth,
                'parent': parent_id,
                'is_split_point': len(edge_index[1][edge_index[0]==cur_id]) > 1,
                'child_num': len(edge_index[1][edge_index[0]==cur_id]),
                'is_leaf': len(edge_index[1][edge_index[0]==cur_id]) == 0
                }
            for child_id in edge_index[1][edge_index[0]==cur_id]:
                stack.append((int(child_id), cur_depth+1, cur_id))
    return visited

# For NeuroMorpho cell type
# for file_name in ['neuro_cell_human', 'neuro_cell_mouse', 'neuro_cell_droso', 'neuro_cell_rat']:

# For NeuroMorpho experimental condition
# for file_name in ['neuro_5x_glia', 'neuro_5x_pc', 'neuro_lps_glia', 'neuro_lps_inter', 'neuro_lps_pc']:

# For NHD watershed data
#for file_name in ['nhd_cluster', 'nhd_diameter', 'nhd_radius']:
for file_name in ['nhd_radius']:
    with open(f'/scratch1/zzha792/nhd/{file_name}.pkl', 'rb') as file:
        dataset = pkl.load(file)
    new_dataset = []
    for graph in tqdm(dataset):
        outgoing_edge_index = graph.edge_index#[:,::2]
        tree = BFS(outgoing_edge_index)
        num_childs = [tree[key]['child_num'] for key in range(len(tree))]   
        depths = [tree[key]['depth'] for key in range(len(tree))]   
        new_data = Data(
            x = graph.x, 
            y = graph.y, 
            num_childs = torch.tensor(num_childs, dtype=torch.float),
            depth = torch.tensor(depths, dtype=torch.float),
            pos = graph.pos,
            edge_index = graph.edge_index
        )
        new_dataset.append(new_data)
    with open(f'/scratch1/zzha792/nhd/{file_name}_tree.pkl', 'wb') as file:
        pkl.dump(new_dataset, file)
