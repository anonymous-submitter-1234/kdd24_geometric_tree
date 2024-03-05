import argparse
import numpy as np
import os
from tqdm import tqdm
import time
import pickle as pkl
import matplotlib.pyplot as plt
import json
import logging

import torch
import copy
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
from torch.optim.lr_scheduler import StepLR

from utils.utils import build_spanning_tree_edge, find_higher_order_neighbors, add_self_loops
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from torch import Tensor
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU, Parameter
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import softmax
import random
from sklearn.metrics import accuracy_score

import torch.nn as nn

class EarthMoversLoss(nn.Module):
    def __init__(self):
        super(EarthMoversLoss, self).__init__()
        
    def forward(self, x, y):
        # Make sure the histograms are normalized
        x = nn.functional.normalize(x, p=1, dim=1)
        y = nn.functional.normalize(y, p=1, dim=1)
        
        # Compute the cumulative sums for each distribution
        cdf_x = torch.cumsum(x, dim=1)
        cdf_y = torch.cumsum(y, dim=1)
        
        # Compute Earth Mover's distance (l1 distance between cdfs)
        emd = torch.norm(cdf_x - cdf_y, p=1, dim=1)
        
        return emd.mean()

def nchilds(data):
    return scatter(
            torch.ones(len(data.edge_index[:,::2][0])),
            index = data.edge_index[:,::2][0],
            dim_size = len(data.x),
            reduce = 'add'
        )

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class Distance_map(torch.nn.Module):
    def __init__(self, cutoff, num_gaussians=50):
        super(Distance_map, self).__init__()
        self.readout = 'add'
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        
    def forward(self, pos, edge_index):
        distances = torch.sqrt(((pos[edge_index[0]] - pos[edge_index[1]])**2).sum(dim=-1))
        distances = self.distance_expansion(distances)
        out = scatter(distances, edge_index[0], dim=0, reduce=self.readout, dim_size=len(pos))        
        return out 


import torch
from torch import Tensor
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
class MLP(torch.nn.Module):
    def __init__(self, input_channels_node, hidden_channels, output_channels, readout='add', num_layers=3):
        super(MLP, self).__init__()
        self.readout = readout    
        self.num_layers = num_layers
        self.mlp = ModuleList()
        block = Sequential(
            Linear(input_channels_node, hidden_channels),
            ReLU(),
        )
        self.mlp.append(block)
        for _ in range(self.num_layers-2):
            block = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
            )
            self.mlp.append(block)
        block = Sequential(
            Linear(hidden_channels, output_channels)
        )
        self.mlp.append(block)

    def forward(self, x, batch):
        for i in range(self.num_layers):
            x = self.mlp[i](x)
        x = scatter(x, batch, dim=0, reduce=self.readout)   
        return x    
class CombinedModel(torch.nn.Module):
    def __init__(self, encoder_model, decoder_model):
        super(CombinedModel, self).__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
    
    def forward(self, x, pos, edge_index, batch, if_GTMP=False):
        if if_GTMP:
            node_emb = self.encoder_model.encode(x, pos, batch, edge_index)
        else:
            node_emb = self.encoder_model.encode(x, pos, edge_index, batch)
        output = self.decoder_model(node_emb, batch)
        return output
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--model', type=str, default='SGCN')
    parser.add_argument('--dataset', type=str, default='Neuron')
    parser.add_argument('--dataset_name', type=str, default='human')
    parser.add_argument('--model_name', type=str, default='human')
    parser.add_argument('--split', type=str, default='811')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--readout', type=str, default='add')

    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--random_seed_2', type=int, default=12345)
    parser.add_argument('--label', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--test_per_round', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--num_gaussians', type=int, default=50)
    parser.add_argument('--cutoff', type=float, default=10.0)
    parser.add_argument('--weight_decay_step_size', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.9)
    args = parser.parse_args()

    return args


def load_data(args):
    if args.dataset == 'Neuron':
        if args.dataset_name in ['human','droso','mouse','rat']:
            with open(os.path.join(args.data_dir, f'neuro_cell_{args.dataset_name}_tree.pkl'), 'rb') as file:
                dataset = pkl.load(file)
        elif args.dataset_name == 'all':
            dataset = []
            for dataset_name in ['human','droso','mouse','rat']:
                with open(os.path.join(args.data_dir, f'neuro_cell_{dataset_name}_tree.pkl'), 'rb') as file:
                    dataset += pkl.load(file)
        else:
            raise Exception(f'Dataset name {args.dataset} not recognized.')
        random.shuffle(dataset)   
        train_valid_split = int( int(args.split[0]) / 10 * len(dataset) )
        valid_test_split = int( int(args.split[1]) / 10 * len(dataset) )

    elif args.dataset == 'Neuron_exp_cond':
        if args.dataset_name in ['5x_glia', '5x_pc', 'lps_glia', 'lps_inter', 'lps_pc']:
            with open(os.path.join(args.data_dir, f'neuro_{args.dataset_name}_tree.pkl'), 'rb') as file:
                dataset = pkl.load(file)
        random.shuffle(dataset)
        train_valid_split = int( int(args.split[0]) / 10 * len(dataset) )
        valid_test_split = int( int(args.split[1]) / 10 * len(dataset) )
    
    else:
        raise Exception(f"Dataset {args.dataset} not recognized.")

    
    train_dataset = dataset[:train_valid_split]
    valid_dataset = dataset[train_valid_split:train_valid_split+valid_test_split]
    test_dataset = dataset[train_valid_split+valid_test_split:]
        
    print('======================')
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of valid graphs: {len(valid_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def main(data, args):
    # device
    if args.device == 'cpu':
        device = torch.device("cpu")
    elif args.device.startswith("cuda"):
        device = torch.device(args.device)
    else:
        raise Exception('Please assign the type of device: cpu or gpu.')


    if args.dataset == 'Neuron':
        input_channels_node, hidden_channels, readout = 4, 64, args.readout
    elif args.dataset == 'Neuron_exp_cond':
        input_channels_node, hidden_channels, readout = 4, 64, args.readout
    num_gaussians = args.num_gaussians
    output_channels = num_gaussians
    num_layers = args.num_layers

    # get train/valid/test data
    train_loader, valid_loader, test_loader = data  

    print(args.model)
    if args.model == 'GIN':
        from models.GIN import GINNet
        net = GINNet(input_channels_node, hidden_channels, output_channels, readout=readout, eps=0., num_layers=args.num_layers)
    elif args.model == 'GCN':
        from models.GCN import GCNNet
        net = GCNNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'PointNetraw':
        from models.PointNetraw import PointNetraw
        net = PointNetraw(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'GAT':
        from models.GAT import GATNet
        net = GATNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'GatedGraphConv':
        from models.GatedGraphConv import GatedNet
        net = GatedNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'PointNet':
        from models.PointNet import PointNet
        net = PointNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'PPFNet':
        from models.PPFNet import PPFNet
        net = PPFNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'SGCN':
        from models.SGCN import SGCN
        net = SGCN(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'Schnet':
        from models.Schnet import Schnet
        net = Schnet(input_channels_node=input_channels_node, 
                    hidden_channels=hidden_channels, output_channels=output_channels, num_interactions=args.num_layers,
                    num_gaussians=hidden_channels, cutoff=args.cutoff, readout=readout)
    elif args.model == 'Dimenet':
        from models.Dimenet import Dimenet
        net = Dimenet(input_channels_node=input_channels_node, 
                    hidden_channels=hidden_channels, output_channels=output_channels, num_blocks=args.num_layers,
                    cutoff=args.cutoff)
    elif args.model == 'SGMP':
        from models.SGMP import SGMP
        net = SGMP(input_channels_node=input_channels_node, 
            hidden_channels=hidden_channels, output_channels=output_channels,
            num_interactions=args.num_layers, cutoff=args.cutoff,
            readout=readout)
    elif args.model == 'GTMP':
        from models.GTMP import GTMP
        net = GTMP(input_channels_node=input_channels_node,
            hidden_channels=hidden_channels, output_channels=output_channels,
            num_interactions=args.num_layers, cutoff=args.cutoff,
            readout=readout)

    model_name = 'model.pt'
    LOG_DIR = os.path.join(args.save_dir, args.dataset, args.model_name, args.model)
    saved_model_file = os.path.join(LOG_DIR, model_name)
    state_dict = torch.load(saved_model_file)
    net.load_state_dict(state_dict)

    LOG_DIR = os.path.join(LOG_DIR, args.dataset_name)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_dir = os.path.join(LOG_DIR, 'finetune.log')
    logging.basicConfig(filename=log_dir, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Start Fine tune.")
    logging.info(f"Config argument:{args}")
    result_file = os.path.join(LOG_DIR, 'fine_tune_results.txt')



    criterion = EarthMoversLoss()#torch.nn.MSELoss()
    model = net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.weight_decay_step_size, gamma=args.weight_decay)
    mlp_model = MLP(hidden_channels, hidden_channels,5)
    decoder_model = CombinedModel(model, mlp_model)
    decoder_criterion = torch.nn.CrossEntropyLoss()
    decoder_model = decoder_model.to(device)
    decoder_optimizer = torch.optim.Adam(decoder_model.parameters(), lr=args.lr)
    decoder_scheduler = StepLR(decoder_optimizer, step_size=args.weight_decay_step_size, gamma=args.weight_decay)
    

    def decoder_train(loader):
        map = Distance_map(cutoff=10.0)
        model.eval()
        decoder_model.train()
        loss_list = []
        for data in (loader):
            x, pos, edge_index, batch, y = data.x.float(), data.pos, data.edge_index, data.batch, data.y
            num_childs, depth = data.num_childs, data.depth
            edge_index = edge_index[:,::2]
            child_distances = map(pos, edge_index)
            x, pos, edge_index, batch, y = x.to(device), pos.to(device), edge_index.to(device), batch.to(device), y.to(device)
            num_childs, depth = num_childs.to(device), depth.to(device)
            child_distances = child_distances.to(device)
            x = torch.cat([num_childs.view(-1,1), depth.view(-1,1), x], dim=-1)
            
            if args.model == 'GTMP':

                edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes, fill_value=-1.)
                _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(edge_index, data.num_nodes, order=3)
                edge_index = edge_index_3rd
                output = decoder_model(x, pos, edge_index, batch, if_GTMP=True)
            else:
                output = decoder_model(x, pos, edge_index, batch)
            
            loss = decoder_criterion(output, y.view(-1))
            loss_list.append(float(loss))
            loss.backward()  # Derive gradients.
            decoder_optimizer.step()  # Update parameters based on gradients.
            decoder_optimizer.zero_grad()  # Clear gradients.
        return np.mean(loss_list)

    def decoder_test(loader):
        map = Distance_map(cutoff=10.0)
        model.eval()
        decoder_model.eval()
        y_pred_list = []
        y_true_list = []
        output_list = []
        softmax = nn.Softmax(dim=1)
        for data in (loader):
            x, pos, edge_index, batch, y = data.x.float(), data.pos, data.edge_index, data.batch, data.y
            num_childs, depth = data.num_childs, data.depth
            edge_index = edge_index[:,::2]
            child_distances = map(pos, edge_index)
            x, pos, edge_index, batch, y = x.to(device), pos.to(device), edge_index.to(device), batch.to(device), y.to(device)
            num_childs, depth = num_childs.to(device), depth.to(device)
            child_distances = child_distances.to(device)
            x = torch.cat([num_childs.view(-1,1), depth.view(-1,1), x], dim=-1)
            if args.model == 'GTMP':

                edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes, fill_value=-1.)
                _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(edge_index, data.num_nodes, order=3)
                edge_index = edge_index_3rd
                output = decoder_model(x, pos, edge_index, batch, if_GTMP=True)
            else:
                output = decoder_model(x, pos, edge_index, batch)
            pred = output.argmax(dim=1)  # Use the class with highest probability.
            y_pred_list += list(pred.cpu().detach().numpy().reshape(-1))
            y_true_list += list(y.cpu().detach().numpy().reshape(-1))

            # normalize output
            out_data = output.detach().cpu().numpy()
            min_values = np.min(out_data, axis=1)
            temp = out_data - min_values[:, np.newaxis]
            row_sums = temp.sum(axis=1)
            normalized_out = temp / row_sums[:, np.newaxis]

            output_list.append(normalized_out)
        return y_pred_list, y_true_list, np.concatenate(output_list, axis=0)

    measure = accuracy_score
    measure_2 = roc_auc_score
    labels = np.arange(5)

    best_model = None
    best_eval_val = 0
    with open(result_file, 'w') as f:
        for epoch in range(1, args.num_epochs+1, 1):
            loss = decoder_train(train_loader)
            y_pred_train, y_true_train, output_train = decoder_test(train_loader) 
            y_pred_valid, y_true_valid, output_valid = decoder_test(valid_loader)
            y_pred_test, y_true_test, output_test = decoder_test(test_loader)
            train_score = measure(y_pred_train, y_true_train)
            valid_score = measure(y_pred_valid, y_true_valid)
            test_score = measure(y_pred_test, y_true_test)
            #print(output_test)
            train_score_2 = measure_2(np.array(y_pred_train), output_train, multi_class='ovo', labels=labels)
            valid_score_2 = measure_2(np.array(y_pred_valid), output_valid, multi_class='ovo', labels=labels)
            test_score_2 = measure_2(np.array(y_pred_test), output_test, multi_class='ovo', labels=labels)
            if valid_score_2 > best_eval_val:
                best_eval_val = valid_score_2
                best_model = copy.deepcopy(decoder_model.state_dict())
                best_scores = [train_score, valid_score, test_score, train_score_2, valid_score_2, test_score_2]
            print('Epoch {:4d}, loss={:.4f}, \
               Train acc: {:.4f}, Valid acc: {:.4f}, Test acc: {:.4f}, \
               Train rocauc: {:.4f}, Valid rocauc: {:.4f}, Test rocauc: {:.4f},'.format(
                   epoch, loss, train_score, valid_score, test_score, train_score_2, valid_score_2, test_score_2
                   )
                , file=f)
            logging.info('Epoch {:4d}, loss={:.4f}, \
               Train acc: {:.4f}, Valid acc: {:.4f}, Test acc: {:.4f}, \
               Train rocauc: {:.4f}, Valid rocauc: {:.4f}, Test rocauc: {:.4f},'.format(
                   epoch, loss, train_score, valid_score, test_score, train_score_2, valid_score_2, test_score_2
                    )
                )
        print('Best results: \
            Train acc: {:.4f}, Valid acc: {:.4f}, Test acc: {:.4f}, \
            Train rocauc: {:.4f}, Valid rocauc: {:.4f}, Test rocauc: {:.4f},'.format(
                best_scores[0], best_scores[1], best_scores[2], best_scores[3], best_scores[4], best_scores[5]
                )
             , file=f)
    model_name = 'finetuned_model.pt'
    saved_model_file = os.path.join(LOG_DIR, model_name)
    torch.save(best_model, saved_model_file)

    
if __name__ == '__main__':
    args = get_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    torch.manual_seed(args.random_seed)  
    data = load_data(args)
    
    main(data, args)
