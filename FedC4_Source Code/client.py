import torch
import torch.nn as nn
from flcore.base import BaseClient
from flcore.fedgcond.fedgc_config import config
from flcore.fedgc.pge import PGE
from torch_geometric.utils.sparse import to_torch_sparse_tensor, dense_to_sparse
import random
import numpy as np
from torch_sparse import SparseTensor
from flcore.fedgc.utils import match_loss, regularization, tensor2onehot
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
import numpy as np
from data.simulation import get_subgraph_pyg_data
from torch_geometric.data import Data
from torch_geometric.utils import to_edge_index
from flcore.fedgc.utils import is_sparse_tensor, normalize_adj_tensor, accuracy
from model.gcn import GCN_kipf
from utils.metrics import compute_supervised_metrics
from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling
import copy
from sklearn.model_selection import train_test_split
from torch_geometric.utils import subgraph
from torch.nn.functional import kl_div, log_softmax, softmax
from scipy.stats import wasserstein_distance

class GCNGenerator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNGenerator, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc_output = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc_output(x).clone()
        return x

class LearnablePoolingLayer(torch.nn.Module):
    def __init__(self, input_dim, pool_ratio=0.5):
        super(LearnablePoolingLayer, self).__init__()
        self.pool = SAGPooling(input_dim, ratio=pool_ratio)

    def forward(self, x, edge_index):
        result = self.pool(x, edge_index)
        pooled_x, pooled_edge_index = result[0], result[1]  # 解包前两个值
        return pooled_x, pooled_edge_index

class BestTracker:
    def __init__(self):
        self.best_accuracy = 0.0
        self.best_round = 0

    def update(self, current_accuracy, current_round):
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.best_round = current_round

    def print_best(self):
        print(f"Best Accuracy: {self.best_accuracy:.4f}, Achieved at Epoch: {self.best_round}")

class FedGcondClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedGcondClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        # print(f"Client {self.client_id} initialized")
        self.mended_graph = None
        self.class_dict2 = None
        self.samplers = None

        self.uncompressed_graph_x = data['x']
        self.uncompressed_graph_edge_index = data['edge_index']
        self.uncompressed_graph_y = data['y']
        self.uncompressed_graph_adj = self.convert_edge_index_to_sparse(self.uncompressed_graph_edge_index, num_nodes=self.uncompressed_graph_x.size(0))
        self.use_compressed_graph=0
        self.task.load_custom_model(GCN_kipf(nfeat=self.task.num_feats,
                                             nhid=args.hid_dim,
                                             nclass=self.task.num_global_classes,
                                             nlayers=args.num_layers,
                                             dropout=args.dropout,
                                             lr=args.lr,
                                             weight_decay=args.weight_decay,
                                             device=self.device))
        print(f"client {self.client_id} graph condensation")
        self.fedgc_initialization()
        self.task.override_evaluate = self.get_override_evaluate()
        

        self.model_optimizer = torch.optim.Adam(self.task.model.parameters(), lr=1e-3, weight_decay=5e-4)

        self.update_compressed_graphs()
        self.notify_server_initialization_complete()

    def execute(self):

        best_tracker = BestTracker()
        
        input_x, input_adj, input_y = self.select_baseline(baseline="ours")
        # input_x = self.add_laplace_noise(input_x, scale=1)

        adj_sparse = to_torch_sparse_tensor(self.uncompressed_graph_edge_index)
        self.synchronize_with_server()

        loss_fn = torch.nn.CrossEntropyLoss()

        num_nodes = self.uncompressed_graph_x.shape[0]

        train_ratio = 0.7

        torch.manual_seed(2025)
        indices = torch.randperm(num_nodes)

        train_size = int(train_ratio * num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        test_mask[indices[train_size:]] = True
        

        for epoch in range(3):
            
            self.task.model.train()
            self.model_optimizer.zero_grad()
            output_train = self.task.model(input_x, input_adj)
            output_val = self.task.model(self.uncompressed_graph_x, self.uncompressed_graph_adj)
            loss_val = loss_fn(output_val[train_mask], self.uncompressed_graph_y[train_mask].detach())
            loss_val.backward()
            self.model_optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                acc_train = accuracy(output_train, input_y.detach())
                print(f"Client {self.client_id} Epoch {epoch + 1}, Loss: {loss_val.item()}, Acc: {acc_train.item()}")
                
            embeddings = self.get_node_embeddings(input_x=input_x, input_adj=input_adj)
            self.task.model.eval()
            with torch.no_grad():
                output = self.task.model(self.uncompressed_graph_x, self.uncompressed_graph_adj)
                predicted_labels = torch.argmax(output, dim=1)
                loss_test = loss_fn(output, self.uncompressed_graph_y)
                acc_test = accuracy(output[test_mask], self.uncompressed_graph_y[test_mask])
                best_tracker.update(acc_test.item(), epoch + 1)

        best_tracker.print_best()
        
        self.message_pool[f"client_{self.client_id}"] = {
        "node_features": self.syn_x.clone()
        }

    def send_message(self):
        node_count = self.syn_x.shape[0]
        with torch.no_grad():
            embeddings = self.task.model.get_embeddings(self.syn_x, self.syn_adj)
        node_features = self.syn_x.clone().cpu()

        self.message_pool[f"client_{self.client_id}"] = {
            "weight": [param.data.clone() for param in self.task.model.parameters()],
            "node_count": node_count,
            "embeddings": embeddings.clone().cpu(),
            "node_features": node_features,
            "labels": self.syn_y.clone().cpu()
        }

        self.uploaded = True
        # print(f"Client {self.client_id} sent weights and node_count: {node_count}")

    def has_uploaded_weights(self):
        return self.uploaded

    def fedgc_initialization(self):
        # conduct local subgraph condensation
        self.num_real_train_nodes = self.task.splitted_data["train_mask"].sum()
        # print(f"Client {self.client_id} has {self.num_real_train_nodes} real train nodes")
        self.num_syn_nodes = int(self.num_real_train_nodes * config["reduction_rate"]*5)
        # self.num_syn_nodes = int(self.num_real_train_nodes * 0.4)
        # trainable parameters
        self.syn_x = nn.Parameter(torch.FloatTensor(self.num_syn_nodes, self.task.num_feats).to(self.device))
        self.pge = PGE(nfeat=self.task.num_feats, nnodes=self.num_syn_nodes, device=self.device, args=self.args).to(self.device)
        self.syn_adj = self.pge(self.syn_x)

        # sampled syn labels
        self.syn_y = torch.LongTensor(self.generate_labels_syn(self.task.splitted_data)).to(self.device)

        # initialize trainable parameters and create optimizer
        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.syn_x], lr=config["lr_feat"]) # parameterized syn_x
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=config["lr_adj"]) # pge: mapping for syn_x -> syn_adj
        
        # print shape
        print('syn_adj:', (self.num_syn_nodes, self.num_syn_nodes), 'syn_x:', self.syn_x.shape)
        self.condense(self.task.splitted_data)
              
    def reset_parameters(self):
        self.syn_x.data.copy_(torch.randn(self.syn_x.size()))

    def generate_labels_syn(self, splitted_data):
        from collections import Counter
        labels_train = splitted_data["data"].y[splitted_data["train_mask"]]
        labels_train = labels_train.tolist()
        counter = Counter(labels_train)
        num_class_dict = {}
        num_train = len(labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(num_train * config["reduction_rate"]*5) - sum_
                # num_class_dict[c] = int(num_train * 0.4) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * config["reduction_rate"]*5), 1)
                # num_class_dict[c] = max(int(num * 0.4), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn
    
    def get_syn_x_initialize(self, splitted_data):
        idx_selected = []
        from collections import Counter;
        counter = Counter(self.syn_y.cpu().numpy())
        for c in range(self.task.num_global_classes):
            # sample 'counter[c]' nodes in class 'c'
            tmp = ((splitted_data["data"].y == c) & splitted_data["train_mask"]).nonzero().squeeze().tolist()
            if type(tmp) is not list:
                tmp = [tmp]
            random.shuffle(tmp)
            tmp = tmp[:counter[c]]
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        idx_selected = torch.LongTensor(idx_selected).to(splitted_data["data"].x.device)
        sub_x = splitted_data["data"].x[idx_selected]

        return sub_x

    def condense(self, splitted_data, verbose=True):
        syn_x, pge, syn_y = self.syn_x, self.pge, self.syn_y

        real_x, real_adj, real_y = splitted_data["data"].x, to_torch_sparse_tensor(splitted_data["data"].edge_index), splitted_data["data"].y
        syn_class_indices = self.syn_class_indices
        if is_sparse_tensor(real_adj):
            real_adj_norm = normalize_adj_tensor(real_adj, sparse=True)
        else:
            real_adj_norm = normalize_adj_tensor(real_adj)

        real_adj_norm = SparseTensor(row=real_adj_norm._indices()[0], col=real_adj_norm._indices()[1],
                value=real_adj_norm._values(), sparse_sizes=real_adj_norm.size()).t()

        # initialize the syn graph with sampled real nodes
        real_x_sub = self.get_syn_x_initialize(splitted_data)

        # Select real nodes as the feature matrix.
        # self.syn_x.data.copy_(real_x_sub)
        
        # hyper-parameters
        outer_loop, inner_loop = self.get_loops()
        loss_avg = 0
         

        # train
        for it in range(config["num_condensation_epochs"]+1):
            self.task.load_custom_model(GCN_kipf(nfeat=self.task.num_feats, 
                                             nhid=self.args.hid_dim, 
                                             nclass=self.task.num_global_classes, 
                                             nlayers=self.args.num_layers, 
                                             dropout=self.args.dropout, 
                                             lr=self.args.lr, 
                                             weight_decay=self.args.weight_decay, 
                                             device=self.device))
            # one-shot
            self.task.model.initialize()
            model_parameters = list(self.task.model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=self.args.lr)
            self.task.model.train()

            

            for ol in range(outer_loop):
                adj_syn = pge(self.syn_x)
                adj_syn_norm = normalize_adj_tensor(adj_syn, sparse=False)

                BN_flag = False
                for module in self.task.model.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    self.task.model.train() # for updating the mu, sigma of BatchNorm
                    output_real = self.task.model.forward(real_x, real_adj_norm)
                    for module in self.task.model.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer
                  
                loss = torch.tensor(0.0).to(self.device)
                
                # class-wise
                for c in range(self.task.num_global_classes):
                    # real loss
                    
                    batch_size, n_id, adjs = self.retrieve_class_sampler(c, real_adj_norm, args=self.args)

                    if n_id == None:
                        
                        continue
                    if self.args.num_layers == 1:
                        adjs = [adjs]
                    
                    adjs = [adj.to(self.device) for adj in adjs]
                    if n_id is None or len(n_id) == 0:
                        print(f"Skipping class {c} as it has no nodes.")
                        continue
                    if real_x[n_id].size(0) == 0:
                        print(f"real_x[n_id] is empty for class {c}.")
                        continue
                    if adjs is None or len(adjs) == 0:
                        print(f"adjs is empty for class {c}.")
                        continue
                    output = self.task.model.forward_sampler(real_x[n_id], adjs)
                    loss_real = F.nll_loss(output, real_y[n_id[:batch_size]])
                      
                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    
                    # syn loss
                    output_syn = self.task.model.forward(syn_x, adj_syn_norm)
                    ind = syn_class_indices[c]
                    loss_syn = F.nll_loss(
                            output_syn[ind[0]: ind[1]],
                            syn_y[ind[0]: ind[1]])
                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                      
                    
                    # gradient match
                    
                    # print(max(self.num_class_dict.values()))
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                     
                    loss += coeff  * match_loss(gw_syn, gw_real, config["dis_metric"], device=self.device)
                     
                
                loss_avg += loss.item()
                  
                # TODO: regularize
                if config["alpha"] > 0:
                    loss_reg = config["alpha"]* regularization(adj_syn, tensor2onehot(syn_y))
                else:
                    loss_reg = torch.tensor(0)

                loss = loss + loss_reg

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                if it % 5 < 2:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                if ol % 5 == 0:
                    # print('Gradient matching loss:', loss.item())
                    continue

                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break

                syn_x_inner = syn_x.detach()
                adj_syn_inner = pge.inference(syn_x_inner)
                adj_syn_inner_norm = normalize_adj_tensor(adj_syn_inner, sparse=False)

                
                
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = self.task.model.forward(syn_x_inner, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, syn_y)
                    loss_syn_inner.backward()
                    # print(loss_syn_inner.item())
                    optimizer_model.step() # update gnn param


            loss_avg /= (self.task.num_global_classes * outer_loop)
            if it % 50 == 0:
                # print('Epoch {}, loss_avg: {}'.format(it, loss_avg))
                continue

            # loss_avg = 0
                
    def get_loops(self):
        # Get the two hyper-parameters of outer-loop and inner-loop.
        # The following values are empirically good.
        if config["one_step"]:
            if self.args.dataset =='ogbn-arxiv':
                return 5, 0
            return 1, 0
        if self.args.dataset in ['Cora']:
            return 20, 10 # sgc
        if self.args.dataset in ['CiteSeer']:
            return 20, 15
        if self.args.dataset in ['Physics']:
            return 20, 10
        else:
            return 20, 10

    def retrieve_class_sampler(self, c, adj, num=64, args=None):
        adj = adj.to(self.device)  # 确保将邻接矩阵移到指定设备

        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.task.num_global_classes):
                idx_mask = (self.task.splitted_data["data"].y == i) & self.task.splitted_data["train_mask"]
                idx = idx_mask.nonzero().squeeze().tolist()
                if type(idx) is not list:
                    idx = [idx]
                self.class_dict2[i] = idx
        
        if args.num_layers == 1:
            sizes = [15]
        elif args.num_layers == 2:
            sizes = [10, 5]
        elif args.num_layers == 3:
            sizes = [15, 10, 5]
        elif args.num_layers == 4:
            sizes = [15, 10, 5, 5]
        elif args.num_layers == 5:
            sizes = [15, 10, 5, 5, 5]

        if self.samplers is None:
            self.samplers = []
            for i in range(self.task.num_global_classes):
                node_idx = torch.LongTensor(self.class_dict2[i]).to(self.device)  # Ensure on correct device
                if node_idx.shape[0] == 0:
                    self.samplers.append(None)
                else:
                    self.samplers.append(NeighborSampler(
                        adj,  # Use the SparseTensor adj directly
                        node_idx=node_idx,
                        sizes=sizes,
                        batch_size=num,
                        num_workers=1,
                        return_e_id=False,
                        num_nodes=adj.size(0),
                        shuffle=True
                    ))
        
        batch = np.random.permutation(self.class_dict2[c])[:num]
        batch = torch.LongTensor(batch).to(self.device)  # Ensure batch is on the same device as adj
        

        batch = self.filter_valid_nodes(batch, adj)

        if self.samplers[c] is not None:

            try:
                out = self.samplers[c].sample(batch)
                return out
            except Exception as e:
                print(f"Error during sampling for class {c}: {e}")
                return None, None, None
        else:
            return None, None, None
        
    def test_with_val(self, verbose=True):
        args=self.args
        res = []
        syn_x, pge, syn_y = self.syn_x.detach(), self.pge, self.syn_y

        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        model = GCN_kipf(nfeat=self.task.num_feats,
                                             nhid=args.hid_dim,
                                             nclass=self.task.num_global_classes,
                                             nlayers=args.num_layers,
                                             dropout=args.dropout,
                                             lr=args.lr,
                                             weight_decay=args.weight_decay,
                                             device=self.device)

        if self.args.dataset in ['ogbn-arxiv']:
            model = GCN_kipf(nfeat=self.task.num_feats,
                                             nhid=args.hid_dim,
                                             nclass=self.task.num_global_classes,
                                             nlayers=args.num_layers,
                                             dropout=args.dropout,
                                             lr=args.lr,
                                             weight_decay=args.weight_decay,
                                             device=self.device)

        adj_syn = pge.inference(syn_x)
        args = self.args

        if self.args.debug:
            torch.save(adj_syn, f'./saved_ours/adj_{args.dataset}_{config["reduction_rate"]}_{args.seed}.pt')
            torch.save(syn_x, f'./saved_ours/feat_{args.dataset}_{config["reduction_rate"]}_{args.seed}.pt')

        if config["lr_adj"] == 0:
            n = len(syn_y)
            adj_syn = torch.zeros((n, n))

        labels_test = self.task.splitted_data["data"].y[self.task.splitted_data["test_mask"]]
        labels_train = self.task.splitted_data["data"].y[self.task.splitted_data["train_mask"]]
        feat_train = self.task.splitted_data["data"].x[self.task.splitted_data["train_mask"]]

        adj_full = to_torch_sparse_tensor(self.task.splitted_data["data"].edge_index)

        train_graph = get_subgraph_pyg_data(self.task.splitted_data["data"],
                                                                 node_list=self.task.splitted_data["train_mask"].nonzero().squeeze().tolist())

        adj_train = to_torch_sparse_tensor(train_graph.edge_index, size=feat_train.shape[0])
        

        model.fit_with_val(syn_x, adj_syn, syn_y, self.task.splitted_data, train_iters=100, normalize=True, verbose=True)

        self.syn_adj = adj_syn
        model.eval()


        output = model.predict(feat_train, adj_train)

        loss_train = F.nll_loss(output, labels_train)
        acc_train = accuracy(output, labels_train)
        if verbose:
            print("Train set results:",
                  "loss= {:.4f}".format(loss_train.item()),
                  "accuracy= {:.4f}".format(acc_train.item()))
        res.append(acc_train.item())

        # Full graph
        output = model.predict(self.task.splitted_data["data"].x, adj_full)
        loss_test = F.nll_loss(output[self.task.splitted_data["test_mask"]], labels_test)
        acc_test = accuracy(output[self.task.splitted_data["test_mask"]], labels_test)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        return res

    def get_override_evaluate(self):
        def override_evaluate(splitted_data=None, mute=False):
            if splitted_data is None:
                splitted_data = self.task.splitted_data
            else:
                names = ["data", "train_mask", "val_mask", "test_mask"]
                for name in names:
                    assert name in splitted_data
            
            
            real_adj = to_torch_sparse_tensor(splitted_data["data"].edge_index)
        
            
            eval_output = {}
            self.task.model.eval()
            with torch.no_grad():
                embedding = None
                logits = self.task.model.forward(splitted_data["data"].x, real_adj)
                loss_train = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["train_mask"])
                loss_val = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["val_mask"])
                loss_test = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["test_mask"])

            
            eval_output["embedding"] = embedding
            eval_output["logits"] = logits
            eval_output["loss_train"] = loss_train
            eval_output["loss_val"]   = loss_val
            eval_output["loss_test"]  = loss_test
            
            
            metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["train_mask"]], labels=splitted_data["data"].y[splitted_data["train_mask"]], suffix="train")
            metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["val_mask"]], labels=splitted_data["data"].y[splitted_data["val_mask"]], suffix="val")
            metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=logits[splitted_data["test_mask"]], labels=splitted_data["data"].y[splitted_data["test_mask"]], suffix="test")
            eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}
            
            info = ""
            for key, val in eval_output.items():
                try:
                    info += f"\t{key}: {val:.4f}"
                except:
                    continue

            prefix = f"[client {self.client_id}]" if self.client_id is not None else "[server]"
            if not mute:
                print(prefix+info)
            return eval_output
        
        return override_evaluate

    def update_compressed_graphs(self):


        self.message_pool[f"client_{self.client_id}"] = {
            "syn_x": self.syn_x,
            "syn_y": self.syn_y,
            "syn_adj": self.syn_adj,
            "num_syn_nodes": self.num_syn_nodes
        }

        # print(f"Client {self.client_id} generated and uploaded compressed graph.")

    def distribute_compressed_graphs(self):

        self.received_compressed_graphs = []
        for client_id in range(self.args.num_clients):
            if f"client_{client_id}" in self.message_pool:
                self.received_compressed_graphs.append(self.message_pool[f"client_{client_id}"])
            else:
                print(f"Client {client_id} has not uploaded compressed graph yet.")

    def notify_server_initialization_complete(self):

        self.message_pool[f"client_{self.client_id}_init_complete"] = True
        # print(f"Client {self.client_id} initialization complete.")

    def adj_matrix_to_edge_list(self, adj_matrix):
        """
        Convert an adjacency matrix to edge list format (2, num_edges).
        """
        edge_index = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()
        return edge_index

    def edge_index_to_adj_matrix(self,  edge_index, num_nodes):
        
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        return adj_matrix

        original_labels = self.uncompressed_graph_y  # 假设未压缩图的标签保存在这里


        new_nodes_count = self.mended_graph["syn_x"].size(0) - original_labels.size(0)


        new_labels = self.generate_labels_for_new_nodes(new_nodes_count)


        combined_labels = torch.cat([original_labels, new_labels], dim=0)

        return combined_labels

        new_labels = torch.randint(0, self.task.num_global_classes, (new_nodes_count,))
        return new_labels

    def synchronize_with_server(self):
        server_weights = self.message_pool["server"]["weight"]
        # print(f"Client {self.client_id} is synchronizing with server.")


        for local_param, global_param in zip(self.task.model.parameters(), server_weights):
            local_param.data = copy.deepcopy(global_param.data)

    def calculate_accuracy(self, output, labels):

        _, predicted = torch.max(output, dim=1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy

    def nodeselctor(self):
        return
    
    def graphrebuilder(self):
        return
        
    def get_node_embeddings(self, input_x, input_adj):

        self.task.model.eval()

        with torch.no_grad():

            embeddings = self.task.model.get_embeddings(input_x, input_adj)
        return embeddings

    def receive_graph_data(self, client_id):

        graph_data = self.message_pool[f"client_{client_id}"]["graph_data"]


        features = graph_data["features"]
        adjacency_matrix = graph_data["adjacency_matrix"]
        labels = graph_data["labels"]


        print(f"Client {client_id} received graph data:")
        print(f"  Features Shape: {features.shape}")
        print(f"  Adjacency Matrix Shape: {adjacency_matrix.shape}")
        print(f"  Labels Shape: {labels.shape}")


        self.features = features
        self.adjacency_matrix = adjacency_matrix
        self.labels = labels

    def convert_edge_index_to_sparse(self, adj_edge_index, num_nodes):

        if adj_edge_index.shape[0] != 2:
            raise ValueError(f"Invalid edge_index shape: {adj_edge_index.shape}. Expected (2, num_edges).")


        row, col = adj_edge_index


        values = torch.ones(row.size(0), dtype=torch.float32, device=adj_edge_index.device)


        sparse_adj = torch.sparse_coo_tensor(
            indices=adj_edge_index,
            values=values,
            size=(num_nodes, num_nodes),
            dtype=torch.float32,
            device=adj_edge_index.device
        )

        return sparse_adj

    def filter_valid_nodes(self, batch, adj):

        valid_batch = []
        for node in batch:
            try:

                if isinstance(node, torch.Tensor):
                    node = node.item()


                if adj[node].nnz() > 0:
                    valid_batch.append(node)
                
            except Exception as e:
                continue

        return torch.LongTensor(valid_batch).to(self.device)
    def select_baseline(self, baseline):
        
        graph_data = self.message_pool.get(f"client_{self.client_id}", {})
        
        if baseline == "random":
            total_nodes = self.uncompressed_graph_x.shape[0]
            random_compression_ratio = 0.001  # Compression ratio
            compressed_size = int(total_nodes * random_compression_ratio)

            # Randomly select node indices
            torch.manual_seed(2024)  # Fix random seed for reproducibility
            selected_indices = torch.randperm(total_nodes)[:compressed_size]


            input_x = self.uncompressed_graph_x[selected_indices]
            input_y = self.uncompressed_graph_y[selected_indices]

            selected_set = set(selected_indices.tolist())
            mask_edge = torch.tensor([
                edge[0].item() in selected_set and edge[1].item() in selected_set
                for edge in self.uncompressed_graph_edge_index.T
            ])
            selected_edges = self.uncompressed_graph_edge_index[:, mask_edge]

            if selected_edges.shape[1] == 0:

                input_adj = to_torch_sparse_tensor(
                    torch.empty((2, 0), dtype=torch.long),
                    size=(compressed_size, compressed_size)
                )
            else:

                mapping = {idx.item(): i for i, idx in enumerate(selected_indices)}
                mapped_edges = torch.stack([
                    torch.tensor([mapping[edge[0].item()], mapping[edge[1].item()]])
                    for edge in selected_edges.T
                ], dim=1)


                input_adj = to_torch_sparse_tensor(mapped_edges, size=(compressed_size, compressed_size))
            return input_x, input_adj, input_y
        
        elif baseline == "fedavg":
            input_x = self.uncompressed_graph_x.detach()
            input_adj = self.uncompressed_graph_adj
            input_y = self.uncompressed_graph_y.detach()
            return input_x, input_adj, input_y
        
        elif baseline == "herding":
            num_nodes = self.uncompressed_graph_x.shape[0]
            selected_size = int(num_nodes * 0.001)


            selected_nodes = [np.random.choice(num_nodes)]
            remaining_nodes = list(set(range(num_nodes)) - set(selected_nodes))


            for _ in range(1, selected_size):
                distances = []
                for node in remaining_nodes:

                    dist = torch.norm(self.uncompressed_graph_x[node] - self.uncompressed_graph_x[selected_nodes], dim=1).min()
                    distances.append(dist.item())
                

                farthest_node = remaining_nodes[np.argmax(distances)]
                selected_nodes.append(farthest_node)
                remaining_nodes.remove(farthest_node)


            selected_nodes = torch.tensor(selected_nodes)


            input_x = self.uncompressed_graph_x[selected_nodes]
            input_y = self.uncompressed_graph_y[selected_nodes]


            if torch.is_tensor(self.uncompressed_graph_adj) and self.uncompressed_graph_adj.is_sparse:

                dense_adj = self.uncompressed_graph_adj.to_dense()
                selected_adj = dense_adj[selected_nodes][:, selected_nodes]
                selected_adj = selected_adj.to_sparse()
            else:

                selected_adj = self.uncompressed_graph_adj[selected_nodes][:, selected_nodes]
            # print(f"Selected adj shape: {selected_adj.shape}")
            # print(f"input_x shape: {input_x.shape}")
            # print(f"input_y shape: {input_y.shape}")
            return input_x, selected_adj, input_y
            
        elif baseline == "coarsening":
            num_nodes = self.uncompressed_graph_x.shape[0]
    

            selected_size = int(num_nodes * 0.001)
            

            torch.manual_seed(2025)
            selected_nodes = torch.randperm(num_nodes)[:selected_size].tolist()


            input_x = self.uncompressed_graph_x[selected_nodes]
            input_y = self.uncompressed_graph_y[selected_nodes]


            dense_adj = self.uncompressed_graph_adj.to_dense()


            selected_nodes_set = set(selected_nodes)
            mask_edge = torch.tensor([ 
                edge[0].item() in selected_nodes_set and edge[1].item() in selected_nodes_set
                for edge in self.uncompressed_graph_adj.T
            ])


            input_adj = dense_adj[selected_nodes, :][:, selected_nodes]
    
            return input_x, input_adj, input_y
        
        elif baseline == "gcond":
            
            input_x = self.syn_x.detach().clone()
            input_adj = self.syn_adj.detach().clone()
            input_y = self.syn_y.detach().clone()
            return input_x, input_adj, input_y
        
        elif baseline == "gcond_x":
            input_x = self.syn_x.detach().clone()
            num_nodes = input_x.shape[0]
            input_adj = torch.eye(num_nodes).detach().clone()
            input_y = self.syn_y.detach().clone()
            return input_x, input_adj, input_y
        
        elif baseline == "whole":
            input_x = self.uncompressed_graph_x.detach()
            input_adj = self.uncompressed_graph_adj
            input_y = self.uncompressed_graph_y.detach()
            return input_x, input_adj, input_y
        
        elif baseline == "ours":
            if hasattr(self, "use_compressed_graph") and self.use_compressed_graph>1:
                print(f"Client {self.client_id}: Using rebuilt graph for training.")
                graph_data = self.message_pool.get(f"client_{self.client_id}", {}).get("graph_data", None)

                input_x = graph_data.get("features", None).detach().clone()
                input_adj = graph_data.get("adjacency_matrix", None).detach().clone()
                input_y = graph_data.get("labels", None).detach().clone()
                
            else:

                input_x = self.syn_x.detach().clone()
                input_adj = self.syn_adj.detach().clone()
                input_y = self.syn_y.detach().clone()
                self.use_compressed_graph += 1
                # self.use_compressed_graph = False
                print(f"Client {self.client_id} using compressed graph for training.")
            return input_x, input_adj, input_y
    def add_laplace_noise(self, features, scale):

        laplace = torch.distributions.Laplace(0, scale)
        noise = laplace.sample(features.shape).to(features.device)


        noisy_features = features + noise
        return noisy_features
