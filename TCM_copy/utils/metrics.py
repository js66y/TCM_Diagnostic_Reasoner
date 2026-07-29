
import torch
import math
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelJaccardIndex
from torchmetrics import Precision, Recall, F1Score, JaccardIndex
# class metrics():
#     def __init__(self, device):
#         self.num = 0
#         self.device = device
#         self.m = {"acc": 0, "class_F1": 0, "MAE": 0}
#         self.matric = MultilabelF1Score(num_labels=510, average='macro').to(self.device)

#     def get_metrics(self, result, target):
#         self.num += 1
#         mask = (result >= 0.5).int()
#         self.m["class_F1"] = self.matric(mask, target).item() + self.m["class_F1"]
#         return {
#             "class_F1": self.m["class_F1"]/self.num, 
#             }


class MultiLabelMetrics:
    def __init__(self, device, model, threshold=0.8, num_labels=510, k_list=[5, 10, 20]):
        self.device = device
        self.num_labels = num_labels
        self.k_list = k_list
        self.model = model
        self.threshold = threshold
        
        # 初始化TorchMetrics指标
        # self.precision_metric = MultilabelPrecision(num_labels, average='macro').to(device)
        # self.recall_metric = MultilabelRecall(num_labels, average='macro').to(device)
        # self.f1_metric = MultilabelF1Score(num_labels, average='macro').to(device)
        # self.jaccard_metric = MultilabelJaccardIndex(num_labels).to(device)
        
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
        self.jaccard_metric = JaccardIndex(
            task='multilabel',
            num_labels=num_labels,
            average="micro",
            threshold=threshold
        ).to(device)
        
        # 自定义NDCG累积变量
        self.ndcg_scores = {k: 0.0 for k in k_list}
        self.sample_count = 0

    # def update(self, preds, preds_values, targets, targets_values):
    def update(self, preds, targets):
        """
        更新指标状态
        preds: 模型输出的pred_logits [batch_size, 510]
        preds_values: 模型输出的pred_values [batch_size, 510]
        targets: 分类预测的标签1 [batch_size, 510]
        targets_values: 回归预测的计量标签 [batch_size, 510]
        """
        probs = torch.sigmoid(preds)
        bin_preds = (probs >= self.threshold).int()
        
        if self.model == "train":
            self.f1_metric.update(probs, targets)
            return

        # 更新TorchMetrics指标
        self.precision_metric.update(probs, targets)
        self.recall_metric.update(probs, targets)
        self.f1_metric.update(probs, targets)
        self.jaccard_metric.update(probs, targets)
        
        # 计算并累积NDCG
        batch_size = preds.size(0)
        self.sample_count += batch_size
        
        for k in self.k_list:
            batch_ndcg = self._calc_batch_ndcg(probs, targets, k)
            self.ndcg_scores[k] += batch_ndcg * batch_size

    def _calc_batch_ndcg(self, probs, targets, k):
        """计算整个batch的NDCG@k平均值"""
        batch_size = probs.size(0)
        batch_ndcg = 0.0
        
        for i in range(batch_size):
            # 获取单个样本的预测和标签
            sample_probs = probs[i]
            sample_targets = targets[i].float()
            
            # 按预测概率降序排序
            _, topk_indices = torch.topk(sample_probs, k)
            rel_scores = sample_targets[topk_indices]
            
            # 计算DCG
            dcg = 0.0
            for pos, rel in enumerate(rel_scores, 1):
                dcg += rel / math.log2(pos + 1)
            
            # 计算IDCG
            topk_targets, _ = torch.topk(sample_targets, k)
            idcg = 0.0
            for pos, rel in enumerate(topk_targets, 1):
                idcg += rel / math.log2(pos + 1)
                
            # 避免除以零
            ndcg = dcg / idcg if idcg > 0 else 0.0
            batch_ndcg += ndcg.item()
        
        return batch_ndcg / batch_size

    def compute(self):
        """计算所有指标"""
        if self.model == "train":
            return {"F1": round(self.f1_metric.compute().item(), 3)}
        metrics = {
            "Precision": round(self.precision_metric.compute().item(), 3),
            "Recall": round(self.recall_metric.compute().item(), 3),
            "F1": round(self.f1_metric.compute().item(), 3),
            "Jaccard": round(self.jaccard_metric.compute().item(), 3),
        }
        
        # 添加NDCG指标
        for k in self.k_list:
            metrics[f"NDCG@{k}"] = round(self.ndcg_scores[k] / self.sample_count, 3)
        return metrics

    def reset(self):
        """重置所有指标"""
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        self.jaccard_metric.reset()
        self.ndcg_scores = {k: 0.0 for k in self.k_list}
        self.sample_count = 0



# import torch
# import math
# try:
#     from torchmetrics import Precision, Recall, F1Score, JaccardIndex
#     TORCHMETRICS_NEW_API = True
# except ImportError:
#     from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelJaccardIndex
#     TORCHMETRICS_NEW_API = False

# class CustomJaccardIndex:
#     """自定义Jaccard指数计算，专门处理稀疏多标签数据"""
#     def __init__(self, num_classes, average='macro', threshold=0.97, ignore_empty_labels=True):
#         self.num_classes = num_classes
#         self.average = average
#         self.threshold = threshold
#         self.ignore_empty_labels = ignore_empty_labels
#         self.device = None  # 动态检测设备
#         self.reset()
    
#     def reset(self):
#         """重置累积状态"""
#         # 初始化时不指定设备，在第一次update时动态设置
#         self.total_intersection = None
#         self.total_union = None
#         self.num_samples = 0
#         self.device = None
    
#     def _ensure_device(self, tensor):
#         """确保累积张量与输入张量在同一设备上"""
#         if self.device is None:
#             self.device = tensor.device
#             self.total_intersection = torch.zeros(self.num_classes, device=self.device)
#             self.total_union = torch.zeros(self.num_classes, device=self.device)
#         elif self.device != tensor.device:
#             # 如果设备发生变化，移动累积张量
#             self.device = tensor.device
#             self.total_intersection = self.total_intersection.to(self.device)
#             self.total_union = self.total_union.to(self.device)
    
#     def update(self, preds, targets):
#         """更新Jaccard指数状态"""
#         # 确保累积张量与输入张量在同一设备上
#         self._ensure_device(preds)
        
#         # 确保输入是二值化的
#         if preds.dtype != torch.long:
#             preds = (preds >= self.threshold).long()
        
#         batch_size = preds.size(0)
#         self.num_samples += batch_size
        
#         # 计算每个标签的交集和并集
#         for i in range(self.num_classes):
#             pred_i = preds[:, i]
#             target_i = targets[:, i]
            
#             intersection = (pred_i * target_i).sum()
#             union = ((pred_i + target_i) > 0).sum()
            
#             self.total_intersection[i] += intersection
#             self.total_union[i] += union
    
#     def compute(self):
#         """计算最终的Jaccard指数"""
#         if self.total_intersection is None:
#             return torch.tensor(0.0)
            
#         jaccard_scores = []
        
#         for i in range(self.num_classes):
#             if self.total_union[i] == 0:
#                 if self.ignore_empty_labels:
#                     # 跳过没有正样本的标签
#                     continue
#                 else:
#                     # 如果都是0，认为完全匹配
#                     jaccard_i = 1.0
#             else:
#                 jaccard_i = self.total_intersection[i] / self.total_union[i]
            
#             jaccard_scores.append(jaccard_i)
        
#         if len(jaccard_scores) == 0:
#             return torch.tensor(0.0, device=self.device)
        
#         if self.average == 'macro':
#             return torch.tensor(sum(jaccard_scores) / len(jaccard_scores), device=self.device)
#         elif self.average == 'micro':
#             total_intersection = self.total_intersection.sum()
#             total_union = self.total_union.sum()
#             if total_union == 0:
#                 return torch.tensor(1.0, device=self.device)
#             return total_intersection / total_union
#         else:
#             return torch.tensor(jaccard_scores, device=self.device)

# class MultiLabelMetrics:
#     def __init__(self, device, model, num_labels=510, k_list=[5, 10, 20]):
#         self.device = device
#         self.model = model
#         self.num_labels = num_labels
#         self.k_list = k_list
        
#         # 初始化TorchMetrics指标
#         if TORCHMETRICS_NEW_API:
#             self.precision_metric = Precision(
#                 task='multilabel', 
#                 num_classes=num_labels,
#                 num_labels=num_labels,
#                 average='macro',
#                 mdmc_reduce='global'
#             ).to(device)
#             self.recall_metric = Recall(
#                 task='multilabel', 
#                 num_classes=num_labels,
#                 num_labels=num_labels,
#                 average='macro',
#                 mdmc_reduce='global'
#             ).to(device)
#             self.f1_metric = F1Score(
#                 task='multilabel', 
#                 num_classes=num_labels,
#                 num_labels=num_labels,
#                 average='macro',
#                 mdmc_reduce='global'
#             ).to(device)
#             # 使用自定义Jaccard计算替代TorchMetrics
#             self.jaccard_metric = CustomJaccardIndex(
#                 num_classes=num_labels,
#                 average='macro',
#                 threshold=0.5,
#                 ignore_empty_labels=True
#             )
#         else:
#             self.precision_metric = MultilabelPrecision(num_labels=num_labels, average='macro').to(device)
#             self.recall_metric = MultilabelRecall(num_labels=num_labels, average='macro').to(device)
#             self.f1_metric = MultilabelF1Score(num_labels=num_labels, average='macro').to(device)
#             # 使用自定义Jaccard计算替代TorchMetrics
#             self.jaccard_metric = CustomJaccardIndex(
#                 num_classes=num_labels,
#                 average='macro',
#                 threshold=0.5,
#                 ignore_empty_labels=True
#             )
        
#         # 自定义NDCG累积变量
#         self.ndcg_sum = {k: 0.0 for k in k_list}
#         self.sample_count = 0

#     def update(self, preds, targets):
#         """
#         更新指标状态
#         preds: 模型输出logits [batch_size, 510]
#         targets: 真实标签 [batch_size, 510]
#         """
#         probs = torch.sigmoid(preds)
#         bin_preds = (probs >= 0.5).int()
        
#         if self.model == "train":
#             self.f1_metric.update(bin_preds, targets)
#             return

#         # 更新TorchMetrics指标
#         self.precision_metric.update(bin_preds, targets)
#         self.recall_metric.update(bin_preds, targets)
#         self.f1_metric.update(bin_preds, targets)
        
#         # 更新自定义Jaccard指标
#         self.jaccard_metric.update(bin_preds, targets)
        
#         # 计算并累积NDCG
#         batch_size = preds.size(0)
#         self.sample_count += batch_size
        
#         for k in self.k_list:
#             batch_ndcg = self._calc_batch_ndcg(probs, targets, k)
#             self.ndcg_sum[k] += batch_ndcg * batch_size

#     def _calc_batch_ndcg(self, probs, targets, k):
#         """计算批次的平均NDCG@k"""
#         batch_size = probs.size(0)
#         ndcg_scores = []
        
#         for i in range(batch_size):
#             prob_i = probs[i]
#             target_i = targets[i]
            
#             # 获取top-k预测
#             _, top_k_indices = torch.topk(prob_i, k)
            
#             # 计算DCG@k
#             dcg = 0.0
#             for j, idx in enumerate(top_k_indices):
#                 if target_i[idx] == 1:
#                     dcg += 1.0 / math.log2(j + 2)  # j+2 because log2(1) is undefined
            
#             # 计算IDCG@k
#             num_relevant = min(target_i.sum().item(), k)
#             idcg = sum(1.0 / math.log2(j + 2) for j in range(num_relevant))
            
#             # 计算NDCG@k
#             if idcg > 0:
#                 ndcg = dcg / idcg
#             else:
#                 ndcg = 0.0
            
#             ndcg_scores.append(ndcg)
        
#         return sum(ndcg_scores) / len(ndcg_scores)

#     def compute(self):
#         """计算所有指标的最终值"""
#         precision = self.precision_metric.compute()
#         recall = self.recall_metric.compute()
#         f1 = self.f1_metric.compute()
#         jaccard = self.jaccard_metric.compute()
        
#         # 计算平均NDCG
#         avg_ndcg = {}
#         for k in self.k_list:
#             avg_ndcg[f"NDCG@{k}"] = round(self.ndcg_sum[k] / max(self.sample_count, 1), 3)
        
#         return {
#             "Precision": round(precision.item(), 3),
#             "Recall": round(recall.item(), 3),
#             "F1": round(f1.item(), 3),
#             "Jaccard": round(jaccard.item(), 3),
#             **avg_ndcg
#         }

#     def reset(self):
#         """重置所有指标"""
#         self.precision_metric.reset()
#         self.recall_metric.reset()
#         self.f1_metric.reset()
#         self.jaccard_metric.reset()
        
#         self.ndcg_sum = {k: 0.0 for k in self.k_list}
#         self.sample_count = 0
