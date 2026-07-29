import os
import torch
import time
import logging
import wandb
from torch import nn
from parse import parse_args
from torch.optim import Adam
import torch.nn.functional as F
from modules import TCM_Model
from torchinfo import summary
from utils.util import ProgressBar
from utils.metrics_value import MultiLabelMetrics
from dataloader import load_tcm_data, load_pre_train, load_bank
from utils.util import set_logger, print_args, set_seed, freeze_layers, unfreeze_layers


logo = r"""
 ████████╗   ██████╗   ██████╗    ██████╗ 
 ╚══██╔══╝  ██╔════╝   ██╔══██╗   ██╔══██╗
    ██║     ██║        ██║  ██║   ██████╔╝
    ██║     ██║        ██║  ██║   ██╔══██╗
    ██║     ╚██████╗   ██████╔╝   ██║  ██║
    ╚═╝      ╚═════╝   ╚═════╝    ╚═╝  ╚═╝
"""
print(logo)


@torch.no_grad()
def Eval(args, model, data_eval, bank, gat):
    model.eval()
    m = MultiLabelMetrics(args.device, model="eval", threshold=0.5)
    val_progress = ProgressBar("Val", total_epoch=1, iters=len(data_eval), width=25)
    m.reset()  # reset metrics
    for item, input in enumerate(data_eval):
        outputs = model(input[0], bank, gat)
        result = outputs['pred_logits']
        pred_values = outputs['pred_values']
        bceloss = nn.BCEWithLogitsLoss()(result, input[-2].float())
        loss_regression = nn.MSELoss()(pred_values, input[-1].float())
        loss = 0.8*bceloss + 0.2*loss_regression
        m.update(result, pred_values, input[-2].long(), input[-1])
        val_log = m.compute()
        val_log["loss"] = round(loss.item(), 3)
        val_progress.update(item, 1, val_log)
    return val_log

@torch.no_grad()
def Test(args, model, data_test, bank, gat):
    if args.predict:
        Predict(args, model, data_test, bank, gat)
    else:
        print("Start Testing ...")
        model = load_pre_train(model, args.pre_train)
        model.eval()
        m = MultiLabelMetrics(args.device, model="test", threshold=0.5)
        val_progress = ProgressBar("Test", total_epoch=1, iters=len(data_test), width=25)
        m.reset()  # reset metrics
        for item, input in enumerate(data_test):
            outputs = model(input[0], bank, gat)
            result = outputs['pred_logits']
            pred_values = outputs['pred_values']
            bceloss = nn.BCEWithLogitsLoss()(result, input[-2].float())
            loss_regression = nn.MSELoss()(pred_values, input[-1].float())
            loss = 0.8*bceloss + 0.2*loss_regression
            m.update(result, pred_values, input[-2].long(), input[-1])
            val_log = m.compute()
            val_log["loss"] = round(loss.item(), 3)
            val_progress.update(item, 1, val_log)
        return val_log

@torch.no_grad()
def Predict(args, model, data_test, bank, gat):
    print("Start Predict ...")
    model = load_pre_train(model, args.pre_train)
    model.eval()
    for item, input in enumerate(data_test):
        outputs = model(input[0], bank, gat)
        result = F.sigmoid(outputs['pred_logits'])
        result[result >= 0.5] = 1
        result[result < 0.5] = 0
        pred_values = outputs['pred_values']
        for i in range(64):
            print("----------------INPUT", i, "----------------")
            print("年龄：", int(input[0][0][i]), 
                  "\n性别：", "男" if int(input[0][1][i][0]) == 1 else "女", 
                  "\n初步诊断：", args.map["preliminary_symptoms"]["index2name"][int(input[0][1][i][1])],
                  "\n中医诊断：", args.map["TCM_symptoms"]["index2name"][int(input[0][1][i][2])],
                  "\n证型：", args.map["syndrome_pattern"]["index2name"][int(input[0][1][i][3])],
                  "\n治则治法：", [args.map["method"]["index2name"][m] for m in input[0][1][i][4:-1].tolist()]
                  )
            print("---------------OUTPUT", i, "----------------")
            print("推荐药物：", [args.map["herb"]["index2name"][idx] for idx in torch.nonzero(result[i]).squeeze(1).tolist()])
            idx = torch.nonzero(result[i]).squeeze(1)
            print("使用剂量：", pred_values[i][idx].long().tolist())
            print("---------Ground_Truth", i, "----------------")
            print("推荐药物：", [args.map["herb"]["index2name"][idx] for idx in torch.nonzero(input[-2][i]).squeeze(1).tolist()])
            print("使用剂量：", input[-1][i][input[-1][i] != 0].tolist())
        exit(0)
         

def Train(args, model, optimizer, scheduler, data_train, data_eval, bank, gat):
    print("Start Training ...")
    logger = logging.getLogger("Train")
    best_epoch, best_precision = 0, 0.0
    if args.wandb:
        wandb.watch(model, log='all')

    m = MultiLabelMetrics(args.device, model="train", threshold=0.5)
    for epoch in range(1, args.epochs + 1):  # train_per_epoch
        # train
        model.train()
        m.reset()  # reset metrics
        start = time.time()
        for item, input in enumerate(data_train):  # per_batch
            outputs = model(input[0], bank, gat)
            result = outputs['pred_logits']
            pred_values = outputs['pred_values']
            bceloss = nn.BCEWithLogitsLoss()(result, input[-2].float())
            loss_regression = nn.MSELoss()(pred_values, input[-1].float())
            loss = 0.8*bceloss + 0.2*loss_regression
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            m.update(result, pred_values, input[-2].long(), input[-1])
        end = time.time()
        train_log = m.compute()
        train_log["loss"] = round(loss.item(), 3)
        log_txt = f"Train Epoch: {epoch}/{args.epochs} || {int((end - start) // 60)} m {int((end - start) % 60)} s || "+' '.join(f'{k}:{v}' for k, v in train_log.items())
        print(log_txt)
        logger.info(log_txt)
        # unfreeze
        if epoch == args.freeze:
            unfreeze_layers(model, optimizer, args.freeze_layers)
        scheduler.step()
        # eval
        eval_log = Eval(args, model, data_eval, bank, gat)
        # wandb
        if args.wandb:
            wandb.log({"loss": train_log["loss"], "lr": train_log["lr"],
                    "r_F1": train_log["F1"],
                    "e_Precision": eval_log["Precision"], "e_Recall": eval_log["Recall"], "e_F1": eval_log["F1"], "e_Jaccard": eval_log["Jaccard"], 
                    "e_NDCG@5": eval_log["NDCG@5"], "e_NDCG@10": eval_log["NDCG@10"], "e_NDCG@20": eval_log["NDCG@20"]
                    })
        # save
        if eval_log["F1"] >= best_precision:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            best_epoch, best_precision = epoch, eval_log["F1"]
            temp_path = os.path.join(args.log_dir, "Best_state_temp.pt")
            final_path = os.path.join(args.log_dir, "Best_state.pt")
            if os.path.exists(final_path):
                os.remove(final_path)
            torch.save(state, temp_path)
            args.pre_train = final_path
            os.rename(temp_path, final_path)
        # early stop
        else:
            if epoch - best_epoch >= args.early_stop:
                print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}, best precision {best_precision}")
                break
    if args.wandb:
        wandb.finish()



def main():
    print("Starting...")
    # infor
    args = parse_args()
    set_logger(args)
    set_seed(args)
    if not torch.cuda.is_available() or args.device < 0:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.device}")
    # data
    bank = load_bank(args)  # load herb bank
    kg = None
    data_train, data_test, data_eval = load_tcm_data(args)  # TCM
    # model, optimizer
    if args.model_name == "TCM":
        model = TCM_Model(
            dropout=args.dp,
            emb_dim=args.dim,
            cat_cardinalities = [2, args.num_pre_symptoms, args.num_tcm_symptoms, args.num_patterns],
            num_herb = args.num_herb,
            device = args.device,
            mask_assoc_dir=os.path.join(args.data_dir, "binary_matrices_mask"),
            graph_assoc_dir=os.path.join(args.data_dir, "binary_matrices_graph"),
            method_dim=args.num_method,
            max_age=args.num_age,
            ).to(args.device)
        Optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        print("select model...")
        exit(0)


    # loading
    print("Start Loading ...") 
    if args.pre_train and args.train and not args.freeze:  # optimizer and model
        model, Optimizer = load_pre_train(model, args.pre_train, Optimizer)
    elif args.pre_train:  # model
        model = load_pre_train(model, args.pre_train)
    else:
        # model.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None)
        ...
    # print_args
    print_args(args)
    scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=7, gamma=0.8)
    # run
    if args.print_model:  # print model
        for i in data_train:
            input = i
            break
        summary(model=model, input_data=(input[0], bank, kg))
    elif args.train:
        if args.freeze:
            freeze_layers(model, Optimizer, args.freeze_layers)
        Train(args, model, Optimizer, scheduler, data_train, data_eval, bank, kg)
        Test(args, model, data_test, bank, kg)
    elif args.test and args.pre_train:
        Test(args, model, data_test, bank, kg)
    elif args.predict and args.pre_train:
        Predict(args, model, data_test, bank, kg)

if __name__ == "__main__":
    main()
