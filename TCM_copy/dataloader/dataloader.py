import os
import torch
import pickle
import random



def load_bank(args):
    map = {'herb':None,'method':None,'preliminary_symptoms':None,'syndrome_pattern':None,'TCM_symptoms':None} 
    for i in map:
        map_path = os.path.join(args.data_dir, "map", i + '_index_map.pkl')
        with open(map_path, 'rb') as f:
            temp = pickle.load(f)
            map[i] = {"index2name": temp['index2name'], "name2index": {v: k for k, v in temp['index2name'].items()}}
   
    if args.predict:
        args.map = map
    args.num_herb = len(map['herb']['name2index'])                                                                             # 510
    args.num_method = len(map['method']['name2index'])                                                                         # 174
    args.num_pre_symptoms = len(map['preliminary_symptoms']['name2index'])                                                     # 654
    args.num_patterns = len(map['syndrome_pattern']['name2index'])                                                             # 312
    args.num_tcm_symptoms = len(map['TCM_symptoms']['name2index'])                                                             # 207
    args.num_age = 104
    return None 



class TCMDataset(torch.utils.data.Dataset):
    def __init__(self, df, args, device, mode):
        self.device = device
        self.df = df
        self.num_herb = args.num_herb
        self.num_method = args.num_method
        self.num_pre_symptoms = args.num_pre_symptoms
        self.num_patterns = args.num_patterns
        self.num_tcm_symptoms = args.num_tcm_symptoms

    def __getitem__(self, idx):
        return [torch.tensor(self.df[idx][0], dtype=torch.float32).to(self.device),torch.tensor(self.df[idx][1], dtype=torch.long).to(self.device)],\
                torch.tensor(self.df[idx][2], dtype=torch.long).to(self.device),\
                torch.tensor(self.df[idx][3], dtype=torch.float32).to(self.device),\
                torch.abs(torch.tensor(self.df[idx][4], dtype=torch.long)).to(self.device)

    def __len__(self):
        return len(self.df)

def load_tcm_data(args): 
    data_train, data_test, data_eval = [],[],[] 
    args.num_train = args.num_val = args.num_test = 0 
    train_loader = vali_loader = test_loader = None 
    # Training evaluates on the held-out test set after selecting the best
    # validation checkpoint, so it needs the test loader as well.
    if args.test or args.predict or args.train:
        data_path = os.path.join(args.data_dir, "test.pkl")
        with open(data_path, "rb") as f:
            data_test = pickle.load(f)
        test_dataset = TCMDataset(data_test, args, args.device, "test")
        args.num_test = len(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)  #, num_workers=8
    if args.train or args.print_model:
        data_path = os.path.join(args.data_dir, "train.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        if args.val_rate < 1:
            random.shuffle(data)
            split_point = int(len(data) * (1 - args.val_rate)) 
            data_train = data[:split_point]
            data_eval = data[split_point:]
            train_dataset = TCMDataset(data_train, args, args.device, "train")
            vali_dataset = TCMDataset(data_eval, args, args.device, "vali")
            args.num_train = len(train_dataset)
            args.num_val = len(vali_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
            vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=args.batch_size, drop_last=True)
    
    print("train:",args.num_train, "val:", args.num_val, "test:",args.num_test)
    return train_loader, test_loader, vali_loader

def load_pre_train(model, pre_train_path, optimizer=None):
    model_device = next(model.parameters()).device
    state = torch.load(pre_train_path, weights_only=True, map_location=model_device)
    model.load_state_dict(state['state_dict'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
        return model, optimizer
    return model

