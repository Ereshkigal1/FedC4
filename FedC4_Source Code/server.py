from datetime import time

import torch
from torch_geometric.utils import to_torch_sparse_tensor
import torch.nn.functional as F
from flcore.base import BaseServer

from model.gcn import GCN_kipf
from flcore.fedgc.utils import match_loss, regularization, tensor2onehot, accuracy
from flcore.fedgc.utils import is_sparse_tensor, normalize_adj_tensor, accuracy
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.spatial.distance import cosine

class FedGcondServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedGcondServer, self).__init__(args, global_data, data_dir, message_pool, device)

        self.task.load_custom_model(GCN_kipf(nfeat=self.task.num_feats,
                                             nhid=args.hid_dim,
                                             nclass=self.task.num_global_classes,
                                             nlayers=args.num_layers,
                                             dropout=args.dropout,
                                             lr=args.lr,
                                             weight_decay=args.weight_decay,
                                             device=self.device))
        

        self.check_and_distribute()

    def execute(self):

        print(f"Starting weight aggregation for {self.args.num_clients} clients.")
        self.aggregate_weights()
        
        self.broadcaster()
        
        self.clientselctor()
        
        self.execute_selectnode()
        
        adj_matrices, merged_labels, total_loss = self.graphrebuilder()
        
        self.send_graph_data_to_clients(adj_matrices)

        print(f"Aggregation executed for {self.args.num_clients} clients.")

    def distribute_weights(self, aggregated_weights):

        print(f"Distributed aggregated weights to clients.")
        print(f"Clients weights: {aggregated_weights[0]}")
        for client_id in range(self.args.num_clients):
            self.message_pool[f"client_{client_id}"]["weight"] = aggregated_weights
            # print(f"Client {client_id} weights: {self.message_pool[f'client_{client_id}']['weight'][0]} from message pool")

    def send_message(self):
        if hasattr(self, 'aggregated_weights') and self.aggregated_weights is not None:

            self.message_pool["server"] = {
                "weight": self.aggregated_weights
            }
            # print(f"Server sent aggregated weights: {self.message_pool['server']['weight'][0]}")
        else:

            print("No aggregated weights found. Using current model parameters.")
            self.message_pool["server"] = {
                "weight": list(self.task.model.parameters())
            }
            # print(f"Server sent current model weights: {self.message_pool['server']['weight'][0]}")

    def check_and_distribute(self):


        if all(f"client_{i}" in self.message_pool for i in range(self.args.num_clients)):
            print("All clients have uploaded compressed graphs. Starting distribution.")


            for client_id in range(self.args.num_clients):
                self.distribute_compressed_graphs(client_id)
        else:
            print("Waiting for all clients to upload compressed graphs.")

    def distribute_compressed_graphs(self, current_client_id):

        received_compressed_graphs = []
        for client_id in range(self.args.num_clients):
            if client_id != current_client_id:
                if f"client_{client_id}" in self.message_pool:
                    received_compressed_graphs.append(self.message_pool[f"client_{client_id}"])
                else:
                    print(f"Client {client_id} has not uploaded compressed graph yet.")


        # print(received_compressed_graphs[0])
        print(f"Distributing compressed graphs to client {current_client_id}.")

    def aggregate_weights(self):

        num_clients = self.args.num_clients
        total_nodes = 0
        aggregated_weights = None


        for client_key, client_data in self.message_pool.items():
            if isinstance(client_data, dict) and "weight" in client_data and "node_count" in client_data:
                client_weights = client_data["weight"]
                node_count = client_data["node_count"]


                if aggregated_weights is None:

                    aggregated_weights = [torch.zeros_like(param) for param in client_weights]


                for i, param in enumerate(client_weights):
                    aggregated_weights[i] += param * node_count


                total_nodes += node_count

        if total_nodes > 0:
            for i in range(len(aggregated_weights)):
                aggregated_weights[i] /= total_nodes

            self.distribute_weights(aggregated_weights)
            print(f"Aggregated weights distributed for {num_clients} clients.")
            self.aggregated_weights = aggregated_weights
            print(f"Aggregated weights successfully stored.")
        else:
            print("No weights were aggregated.")

    def get_client_embeddings(self):

        client_embeddings = []

        for client_id in range(self.args.num_clients):
            client_data = self.message_pool.get(f"client_{client_id}")
            if client_data:
                embeddings = client_data.get("embeddings", None)
                if embeddings is not None:
                    client_embeddings.append(embeddings)
                    # print(f"Client {client_id} embeddings received: shape {embeddings.shape}")
                else:
                    print(f"Client {client_id} embeddings not found.")
            else:
                print(f"No data received from Client {client_id}.")

        return client_embeddings

    def get_client_features(self):

        client_features = []

        for client_id in range(self.args.num_clients):
            client_data = self.message_pool.get(f"client_{client_id}")
            if client_data:
                features = client_data.get("node_features", None)
                if features is not None:
                    client_features.append(features)
                    # print(f"Client {client_id} features received: shape {features.shape}")
                else:
                    print(f"Client {client_id} features not found.")
            else:
                print(f"No data received from Client {client_id}.")

        return client_features

    def get_client_labels(self):

        client_labels = []

        for client_id in range(self.args.num_clients):
            client_data = self.message_pool.get(f"client_{client_id}")
            if client_data:
                labels = client_data.get("labels", None)
                if labels is not None:
                    client_labels.append(labels)
                    # print(f"Client {client_id} features received: shape {labels.shape}")
                else:
                    print(f"Client {client_id} labels not found.")
            else:
                print(f"No data received from Client {client_id}.")

        return client_labels
    
    def broadcaster(self):
        all_embeddings = self.get_client_embeddings()
        print(f"Total embeddings collected: {len(all_embeddings)}")
        
        distributions = []
        avg_embeddings = []
        num_clients = self.args.num_clients  

        for i, embeddings in enumerate(all_embeddings):

            norms = torch.norm(embeddings, dim=1).cpu().numpy()
            distributions.append(norms)


            avg_embedding = embeddings.mean(dim=0, keepdim=True)  # shape: (1, embedding_dim)
            avg_embeddings.append(avg_embedding)
            
            # print(f"Client {i} Embedding Norm Distribution: {norms}")
            # print(f"Client {i} Average Embedding: {avg_embedding}")
        
        self.broadcast_embeddings_to_clients(distributions, avg_embeddings)


        
    def broadcast_embeddings_to_clients(self, distributions, avg_embeddings):
        num_clients = self.args.num_clients

        for client_id in range(num_clients):
            if f"client_{client_id}" not in self.message_pool:
                self.message_pool[f"client_{client_id}"] = {}

            self.message_pool[f"client_{client_id}"]["embedding_distribution"] = distributions[client_id]
            self.message_pool[f"client_{client_id}"]["average_embedding"] = avg_embeddings[client_id]



    def clientselctor(self, num_groups=2):


        client_distributions = [
            self.message_pool[f"client_{client_id}"]["embedding_distribution"]
            for client_id in range(self.args.num_clients)
        ]


        num_clients = len(client_distributions)
        

        similarity_matrix = np.zeros((num_clients, num_clients))


        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:
                    wd_distance = wasserstein_distance(client_distributions[i], client_distributions[j])
                    similarity_matrix[i, j] = wd_distance
                else:
                    similarity_matrix[i, j] = 0.0
                    
        if np.isnan(similarity_matrix).any():
            print("NaN values found in similarity_matrix. Handling them...")

            similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)



        clustering = AgglomerativeClustering(n_clusters=num_groups, metric='precomputed', linkage='average')
        labels = clustering.fit_predict(similarity_matrix)


        # print("Grouping Results (Cluster Labels):")
        # print(labels)


        for client_id in range(num_clients):
            if f"client_{client_id}" not in self.message_pool:
                self.message_pool[f"client_{client_id}"] = {}
            self.message_pool[f"client_{client_id}"]["group_label"] = labels[client_id]

        return similarity_matrix, labels

    def nodeselctor(self, client_id):
        """
        Select nodes from other clients based on similarity to the Mean-embedding
        of a specified client and merge their labels.
        """
        avg_embeddings = [
            self.message_pool[f"client_{i}"]["average_embedding"]
            for i in range(self.args.num_clients)
        ]

        all_embeddings = self.get_client_embeddings()
        all_node_features = self.get_client_features()
        all_labels = self.get_client_labels()

        clustering_labels = [
            self.message_pool[f"client_{i}"]["group_label"]
            for i in range(self.args.num_clients)
        ]

        target_mean_embedding = avg_embeddings[client_id].squeeze()
        selected_node_features = []
        selected_node_labels = []

        for other_client_id, cluster_label in enumerate(clustering_labels):
            if other_client_id == client_id:
                continue
            if cluster_label == clustering_labels[client_id]:
                other_client_embeddings = all_embeddings[other_client_id]
                other_client_features = all_node_features[other_client_id]
                other_client_labels = all_labels[other_client_id]

                for node_idx in range(len(other_client_embeddings)):
                    node_embedding = other_client_embeddings[node_idx]
                    similarity = 1 - cosine(
                        target_mean_embedding.cpu().numpy(),
                        node_embedding.cpu().numpy(),
                    )
                    
                    if similarity > 0.4:
                        selected_node_features.append(other_client_features[node_idx])
                        selected_node_labels.append(other_client_labels[node_idx])

        if selected_node_features:
            selected_features_tensor = torch.stack(selected_node_features)
            selected_labels_tensor = torch.tensor(selected_node_labels, dtype=torch.long)

            original_features = self.message_pool[f"client_{client_id}"]["node_features"]
            original_labels = self.message_pool[f"client_{client_id}"]["labels"]

            merged_features = torch.cat((original_features, selected_features_tensor), dim=0)
            merged_labels = torch.cat((original_labels, selected_labels_tensor), dim=0)

            self.message_pool[f"client_{client_id}"]["merged_node_features"] = merged_features
            self.message_pool[f"client_{client_id}"]["merged_labels"] = merged_labels

            print(f"Client {client_id} selected {len(selected_node_features)} nodes from other clients in the same cluster.")
            print(f"Merged features shape: {merged_features.shape}, Merged labels shape: {merged_labels.shape}")
        else:

            self.message_pool[f"client_{client_id}"]["merged_node_features"] = self.message_pool[f"client_{client_id}"]["node_features"]
            self.message_pool[f"client_{client_id}"]["merged_labels"] = self.message_pool[f"client_{client_id}"]["labels"]
            print(f"Client {client_id} selected no nodes for merging.")


    def execute_selectnode(self):

        merged_feature_matrices = []

        for client_id in range(self.args.num_clients):

            self.nodeselctor(client_id)


            merged_features = self.message_pool[f"client_{client_id}"]["merged_node_features"]
            merged_labels = self.message_pool[f"client_{client_id}"]["merged_labels"]


            assert merged_features.size(0) == merged_labels.size(0), \
                f"Client {client_id}: Features and labels size mismatch after merging!"


            merged_feature_matrices.append(merged_features)


        self.merged_feature_matrices = merged_feature_matrices


        # print(f"Total merged feature matrices collected: {len(merged_feature_matrices)}")


    def graphrebuilder(self):

        adj_matrices = []
        total_loss = 0.0
        merged_labels = []

        lambda_reconstruction = 1
        lambda_similarity = 500
        lambda_sparsity = 100
        learning_rate = 0.01
        num_iterations = 100

        for client_idx, feature_matrix in enumerate(self.merged_feature_matrices):
            num_nodes = feature_matrix.size(0)
            Z = torch.randn(num_nodes, num_nodes, requires_grad=True, device=feature_matrix.device)  # (N, N)
            optimizer = torch.optim.Adam([Z], lr=learning_rate)

            merged_labels.append(self.message_pool[f"client_{client_idx}"]["merged_labels"])

            for iteration in range(num_iterations):
                optimizer.zero_grad()

                reconstructed_X = torch.matmul(Z, feature_matrix)
                reconstruction_loss = torch.norm(feature_matrix - reconstructed_X, p='fro') ** 2

                normalized_Z = F.normalize(Z, p=2, dim=1)
                similarity_matrix = torch.mm(normalized_Z, normalized_Z.T)
                similarity_loss = torch.sum((1 - similarity_matrix) * torch.abs(Z))

                sparsity_loss = torch.norm(Z, p=1)
                # print(reconstruction_loss)
                # print(similarity_loss)
                # print(sparsity_loss)
                client_loss = (
                    lambda_reconstruction * reconstruction_loss +
                    lambda_similarity * similarity_loss +
                    lambda_sparsity * sparsity_loss
                )
                # print(client_loss)
                client_loss.backward()
                optimizer.step()

                    
                with torch.no_grad():
                    Z.data = F.relu(Z.data)
                    
                with torch.no_grad():
                    Z.data = (Z.data + Z.data.T) / 2


            with torch.no_grad():
                Z.data.fill_diagonal_(0)

                row_sum = Z.sum(dim=1)
                D_inv_sqrt = torch.diag(1.0 / torch.sqrt(row_sum + 1e-10))

                normalized_adj = torch.mm(D_inv_sqrt, torch.mm(Z, D_inv_sqrt))  # D^(-1/2) A D^(-1/2)
                
            adj_matrices.append(normalized_adj.detach().cpu())
            total_loss += client_loss.item()

        self.adj_matrices = adj_matrices
        self.merged_labels = merged_labels

        # print(f"Graph rebuilding complete. Total loss across all clients: {total_loss:.4f}")

        return adj_matrices, merged_labels, total_loss

    def send_graph_data_to_clients(self, adj_matrices):
        for client_idx in range(self.args.num_clients):
            if f"client_{client_idx}" not in self.message_pool:
                self.message_pool[f"client_{client_idx}"] = {}

            feature_matrix = self.merged_feature_matrices[client_idx]
            labels = self.merged_labels[client_idx]
            adjacency_matrix = adj_matrices[client_idx]

            graph_data = {
                "features": feature_matrix.cpu(),
                "adjacency_matrix": adjacency_matrix.cpu(),
                "labels": labels.cpu()
            }
            
            self.message_pool[f"client_{client_idx}"]["graph_data"] = graph_data

            print(f"Sent graph data to Client {client_idx}: "
                f"Features Shape: {feature_matrix.shape}, "
                f"Adjacency Matrix Shape: {adjacency_matrix.shape}, "
                f"Labels Shape: {labels.shape}")
            


         