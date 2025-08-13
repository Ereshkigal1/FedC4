from flcore.trainer import FGLTrainer
from utils.basic_utils import seed_everything
import sys
import warnings
import torch
import time
import faulthandler
faulthandler.enable()
torch.autograd.set_detect_anomaly(True)
torch.cuda.memory_allocated()
warnings.filterwarnings("ignore")


sys.argv = ["main.py",
            '--debug', 'False',
            "--seed", '2025',
            '--scenario', 'subgraph_fl',
            '--simulation_mode', 'subgraph_fl_louvain',
            '--task', 'node_cls',
            '--louvain_resolution', '1',
            '--dataset', 'Cora',
            '--model', 'gcn',
            '--fl_algorithm', 'fedgcond',
            '--num_clients', '5',
            '--num_epochs', '3',
            '--metrics', 'accuracy',
            '--evaluation_mode', "local_model_on_local_data"]

from config import args
if args.seed != 0:
    seed_everything(args.seed)

# print(args)
# set --root to store raw and processed dataset in your own path
trainer = FGLTrainer(args)
trainer.train()
