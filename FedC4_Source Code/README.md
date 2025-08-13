**FedC4** (*Fed*erated Graph Learning with Graph *C*ondensation and *C*lient-*C*lient *C*ollaboration) is an advanced framework designed for Federated Graph Learning (FGL) that integrates Graph Condensation (GC) techniques with Client-Client (C-C) collaboration. This method addresses three key challenges in traditional FGL paradigms:

1. **Fine-Grained Personalization:** FedC4 provides personalized information exchange at a finer granularity, enabling clients to share and utilize tailored knowledge for improved local training.
2. **Communication Overhead:** By integrating GC and selective information sharing, FedC4 significantly reduces the communication cost compared to traditional C-C strategies.
3. **Privacy Preservation:** FedC4 leverages the inherent privacy-preserving properties of GC to protect sensitive data during collaborative learning.

FedC4 achieves state-of-the-art performance across multiple graph-based tasks while maintaining efficiency and privacy.

## Before Started

You can modify the experimental settings in `/config.py` as needed.

### Scenario and Dataset Simulation Settings

```python
--scenario           # fgl scenario
--root               # root directory for datasets
--dataset            # list of used dataset(s)
```

### Communication Settings

```python
--num_clients        # number of clients
--num_rounds         # number of communication rounds
--client_frac        # client activation fraction
```


### Model and Task Settings
```python
--task               # downstream task
--train_val_test     # train/validatoin/test split proportion
--num_epochs         # number of local epochs
--dropout            # dropout
--lr                 # learning rate
--optim              # optimizer
--weight_decay       # weight decay
--model              # gnn backbone
--hid_dim            # number of hidden layer units
```

## Get Started

Our code is built upon the **OpenFGL** framework. To use our implementation, follow these steps:

1. **Download OpenFGL Framework:**  
   Clone and deploy the [OpenFGL](https://github.com/zyl24/OpenFGL) framework as the base code.

2. **Set Up Method Directory:**  
   - Create a new method folder under `flcore`.
   - Copy all Python files (except `main.py`) from this repository into the newly created folder.

3. **Download Datasets:**  
   - Place the required datasets into the `dataset` directory.

4. **Handle Additional Datasets and Baselines:**  
   - OpenFGL does not include all the datasets and baselines we used.  
   - For datasets, add the relevant processing methods in `data/global_dataset_loader.py`.  
   - For baselines, the implementation is included in `client.py`. Select the corresponding baseline when running experiments.

5. **Adjust FGL and GC Parameters:**  
   - Some specific parameters for FGL and GC processes need to be configured in `client.py` and `server.py`. Update these files as needed for your experiments.

By following these steps, you can seamlessly integrate and utilize our FedC4 framework with OpenFGL. For further instructions or issues, please refer to our documentation or contact us directly.


## Cite
Please cite our paper (and the respective papers of the methods used) if you use this code in your own work:
```
```
