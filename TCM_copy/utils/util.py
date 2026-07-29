import os
import sys
import time
import wandb
import torch
import pickle
import shutil
import logging
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from datetime import datetime
from prettytable import PrettyTable




def print_args(args, str_num=80):
    # formatter
    for arg, val in args.__dict__.items():
        logger = logging.getLogger()
        logger.info(arg + "." * (str_num - len(arg) - len(str(val))) + str(val))

def set_logger(args):
    # log_dir
    args.log_dir = f"{args.log_dir}/n-{args.model_name}/v-{args.version}/t-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # copy_code
    with open(os.path.join(args.model_dir, "__init__.py"), 'r') as config_file:
        target_directory = config_file.readline().split('.')[1].split()[0]
        shutil.copy2(os.path.join(args.model_dir, target_directory + ".py"), args.log_dir)

    # Formatter
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )

    # FileHandler
    file_handler = logging.FileHandler(f"{args.log_dir}/run.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # StreamHandler
    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.INFO)
    # stream_handler.setFormatter(formatter)
    # logger.addHandler(stream_handler)
    # wandb
    if args.wandb:
        wandb.login()#登录 WandB 账号
        wandb.init(
            project=args.model_name,
            name=args.log_dir.split('/')[-1],
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
            },
        )

def set_seed(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.CurrentPID = os.getpid()
    args.cwd = os.getcwd()

def set_map(args):
    map = {}
    names = ["preliminary_symptoms","TCM_symptoms","syndrome_pattern","method","herb_and_weight"]
    for i in names:
        with open('dataset/'+i+'_index_map.pkl', 'rb') as f:
            map[i] = pickle.load(f)
    args.map = map

def set_device(args):
    if not torch.cuda.is_available() or args.device < 0:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.device}")

class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"

    def __init__(self, mode, epoch=None, total_epoch=None, iters=None, current=None, width=30, symbol="\u2588",
                 output=sys.stdout):  # stdout
        assert len(symbol) == 1
        self.mode = mode
        self.iters = iters
        self.symbol = symbol
        self.output = output
        self.width = width
        self.current = current
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.txt = ""
        self.args = {}
        self.pre_time = 0.0
        self.cur_time = 0.0
        self.total_time = 0.0
        self.started_time = 0.0

    def update(self, current, epoch=None, message: dict = {}):
        self.current = current
        if epoch:
            self.epoch = epoch
        txt = []
        for k, v in message.items():
            if k == "lr":
                txt.append(f'{k}:{v:.1e} ')
            else:
                txt.append(f'{k}:{v:.3f} ')
        self.txt = ''.join(txt)
        self.__call__(message)

    def sec2time(self, sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        pattern = r'%02d:%02d:%02d'
        return pattern % (h, m, s)

    def __call__(self, m):
        percent = self.current / float(self.iters)
        size = int(self.width * percent)
        bar = "\u2502" + self.symbol * size + " " * (self.width - size) + "\u2502"

        self.pre_time = time.time() if self.pre_time == 0 else self.pre_time

        self.cur_time = time.time()
        spend_time = self.cur_time - self.pre_time
        self.pre_time = self.cur_time

        self.started_time += spend_time
        self.total_time = self.started_time + spend_time * (self.iters - self.current)

        self.args.update({
            "mode": self.mode,
            "iters": self.iters,
            "bar": bar,
            "current": self.current,
            "percent": percent,
            "txt": self.txt,
            "epoch": self.epoch,
            "epochs": self.total_epoch,
            "time": f"{spend_time:.4f}",
            "started_time": self.sec2time(self.started_time),
            "total_time": self.sec2time(self.total_time)
        })
        message = "\033[1;31m{mode} Epoch: {epoch}/{epochs}\033[0m \033[1;33m {bar} \033[0m  \033[1;32m[ {txt} ]\033[0m \033[1;36m[ {current}/{iters} | {time} sec/it | {started_time}/{total_time} | {percent:.2%} ]\033[0m".format(
            **self.args)
        if self.current == self.iters - 1:
            logger = logging.getLogger(self.args["mode"])
            if self.args["mode"] == "Train":
                logger.info(str(self.args["epoch"])+": "+self.args["txt"])
            elif self.args["mode"] == "Val":
                logger.info(self.args["txt"])
            else:
                test_metric_header = ["Precision", 
                                      "Recall", "F1", 
                                      "NDCG@5", "MAE", "RMSE",
                                      "loss"]
                test_table = PrettyTable(test_metric_header)
                test_table.align = "l"
                test_table.add_row(m.values())
                logger.info(f"\n{test_table.get_string()}")
            print("\r", message, file=self.output, end="\n")
            self.started_time = 0.0
            self.total_time = 0.0
            self.pre_time = 0.0
        else:
            # print("\r" + message, file=self.output, end="")
            ...

def index2onehot(index, num_classes):
    method_hot = [0] * num_classes
    for i in index:
        method_hot[i] = 1
    return method_hot

def indexs2onehots(indexs, num_classes):
    return [index2onehot(i, num_classes) for i in indexs]



def freeze_layers(model, optimizer, layers_to_freeze):
    for name, param in model.named_parameters():
        if any([layer in name for layer in layers_to_freeze]):
            param.requires_grad = False
    optimizer.param_groups = []
    optimizer.add_param_group({'params': [param for name, param in model.named_parameters() if not any([layer in name for layer in layers_to_freeze])]})

def unfreeze_layers(model, optimizer, layers_to_freeze):
    for name, param in model.named_parameters():
        if any([layer in name for layer in layers_to_freeze]):
            param.requires_grad = True
    optimizer.add_param_group({'params': [param for name, param in model.named_parameters() if any([layer in name for layer in layers_to_freeze])]})





