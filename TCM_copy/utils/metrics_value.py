import torch
from torchmetrics import (
    Precision, 
    Recall, 
    F1Score, 
    JaccardIndex,
    MeanAbsoluteError,
    MeanSquaredError
)

class MultiLabelMetrics:
    def __init__(self, device, model, threshold=0.8, num_labels=510, k_list=[5]):
        self.device = device
        self.num_labels = num_labels
        self.k_list = k_list
        self.model = model
        self.threshold = threshold
        self.precision_metric = Precision(
            task='multilabel',
            num_labels=num_labels,
            average="micro",
            threshold=threshold
        ).to(device)
        self.recall_metric = Recall(
            task='multilabel',
            num_labels=num_labels,
            average="micro",
            threshold=threshold
        ).to(device)
        self.f1_metric = F1Score(
            task='multilabel',
            num_labels=num_labels,
            average="micro",
            threshold=threshold
        ).to(device)
        self.ndcg_scores = {k: 0.0 for k in k_list}
        self.sample_count = 0
        self.mae_metric = MeanAbsoluteError().to(device)
        self.mse_metric = MeanSquaredError().to(device)

    def update(self, preds, preds_values, targets, targets_values):
        probs = torch.sigmoid(preds)
        
        if self.model == "train":
            self.f1_metric.update(probs, targets)
            self.mae_metric.update(preds_values, targets_values)
            return
        self.precision_metric.update(probs, targets)
        self.recall_metric.update(probs, targets)
        self.f1_metric.update(probs, targets)
        batch_size = preds.size(0)
        self.sample_count += batch_size
        for k in self.k_list:
            batch_ndcg = self._calc_batch_ndcg(probs, targets, k)
            self.ndcg_scores[k] += batch_ndcg * batch_size
        self.mae_metric.update(preds_values, targets_values)
        self.mse_metric.update(preds_values, targets_values)


    def _calc_batch_ndcg(self, probs, targets, k):
        batch_size = probs.size(0)
        batch_ndcg = 0.0
        
        for i in range(batch_size):
            sample_probs = probs[i]
            sample_targets = targets[i].float()
            
            _, topk_indices = torch.topk(sample_probs, k)
            rel_scores = sample_targets[topk_indices]
            
            dcg = (rel_scores / torch.log2(torch.arange(2, k + 2, device=self.device))).sum()
            
            topk_targets, _ = torch.topk(sample_targets, k)
            idcg = (topk_targets / torch.log2(torch.arange(2, k + 2, device=self.device))).sum()
                
            ndcg = dcg / idcg if idcg > 0 else 0.0
            batch_ndcg += ndcg.item()
        
        return batch_ndcg / batch_size

    def compute(self):
        if self.model == "train":
            return {
                "F1": round(self.f1_metric.compute().item(), 3),
                "MAE": round(self.mae_metric.compute().item(), 3),
            }
            
        metrics = {
            "Precision": round(self.precision_metric.compute().item(), 3),
            "Recall": round(self.recall_metric.compute().item(), 3),
            "F1": round(self.f1_metric.compute().item(), 3)
        }
        
        if self.sample_count > 0:
            for k in self.k_list:
                metrics[f"NDCG@{k}"] = round(self.ndcg_scores[k] / self.sample_count, 3)

        metrics["MAE"] = round(self.mae_metric.compute().item(), 3)
        mse = self.mse_metric.compute()
        metrics["RMSE"] = round(torch.sqrt(mse).item(), 3)
        
        return metrics

    def reset(self):
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        self.ndcg_scores = {k: 0.0 for k in self.k_list}
        self.sample_count = 0
        
        self.mae_metric.reset()
        self.mse_metric.reset()