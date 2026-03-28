# my_confuse_matrix.py
import torch
import numpy as np
import time
from sklearn.metrics import average_precision_score

class ConfusionMatrix(object):
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = None
        self._extra = {}  # <<< 新增：用于挂载额外指标（如 hd95）
        
        # MAP和FPS相关变量
        self.predictions_for_map = []  # 存储预测概率用于MAP计算
        self.targets_for_map = []     # 存储真实标签用于MAP计算
        self.inference_times = []     # 存储推理时间用于FPS计算

    # <<< 新增：设置/更新额外指标
    def set_extra(self, **kwargs):
        """
        用法：confmat.set_extra(hd95_per_class=[...], hd95_macro=0.1234, note="...") 
        kwargs 的值应该是可序列化/可打印的（list/str/float 等）
        """
        self._extra.update(kwargs)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor, pred_probs: torch.Tensor = None, inference_time: float = None):
        """
        pred: [B,H,W]  int64 - 预测类别
        target: [B,H,W] int64 - 真实标签
        pred_probs: [B,C,H,W] float32 - 预测概率（用于MAP计算）
        inference_time: float - 推理时间（用于FPS计算）
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)

        # 过滤无效像素（含 ignore_index）
        mask = (target != self.ignore_index) & (pred >= 0) & (pred < n) & (target >= 0) & (target < n)
        if not mask.any():
            return

        inds = n * pred[mask].to(torch.int64) + target[mask].to(torch.int64)
        self.mat += torch.bincount(inds, minlength=n * n).reshape(n, n)
        
        # 存储MAP计算所需的数据
        if pred_probs is not None:
            # 将预测概率和目标转换为numpy数组并存储
            pred_probs_np = pred_probs.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            
            # 确保列表长度足够
            while len(self.predictions_for_map) < n:
                self.predictions_for_map.append([])
            while len(self.targets_for_map) < n:
                self.targets_for_map.append([])
            
            # 为每个类别存储数据
            for class_idx in range(n):
                # 获取当前类别的预测概率和二值化目标
                class_probs = pred_probs_np[:, class_idx, :, :].flatten()
                class_targets = (target_np == class_idx).astype(int).flatten()
                
                # 过滤掉ignore_index的像素
                valid_mask = (target_np != self.ignore_index).flatten()
                class_probs = class_probs[valid_mask]
                class_targets = class_targets[valid_mask]
                
                # 只存储有效的数据
                if len(class_probs) > 0 and len(class_targets) > 0:
                    self.predictions_for_map[class_idx].extend(class_probs.tolist())
                    self.targets_for_map[class_idx].extend(class_targets.tolist())
        
        # 存储推理时间
        if inference_time is not None:
            self.inference_times.append(inference_time)

    def compute_map(self):
        """
        计算MAP (Mean Average Precision)
        基于存储的预测概率和目标标签计算每个类别的AP，然后计算平均值
        """
        if not self.predictions_for_map or not self.targets_for_map:
            return torch.zeros(self.num_classes), torch.tensor(0.0)
        
        ap_scores = []
        for class_idx in range(self.num_classes):
            if (class_idx < len(self.predictions_for_map) and 
                class_idx < len(self.targets_for_map) and 
                len(self.predictions_for_map[class_idx]) > 0 and
                len(self.targets_for_map[class_idx]) > 0):
                try:
                    # 检查是否有正样本
                    targets = np.array(self.targets_for_map[class_idx])
                    predictions = np.array(self.predictions_for_map[class_idx])
                    
                    if np.sum(targets) > 0:  # 有正样本
                        # 计算当前类别的Average Precision
                        ap = average_precision_score(targets, predictions)
                        ap_scores.append(ap)
                    else:
                        # 没有正样本，AP设为0
                        ap_scores.append(0.0)
                except (ValueError, Exception) as e:
                    # 如果计算出错，AP设为0
                    ap_scores.append(0.0)
            else:
                ap_scores.append(0.0)
        
        ap_tensor = torch.tensor(ap_scores, dtype=torch.float32)
        map_score = ap_tensor.mean()
        
        return ap_tensor, map_score
    
    def compute_fps(self):
        """
        计算FPS (Frames Per Second)
        基于存储的推理时间计算平均FPS
        """
        if not self.inference_times:
            return torch.tensor(0.0)
        
        # 计算平均推理时间
        avg_inference_time = sum(self.inference_times) / len(self.inference_times)
        
        # FPS = 1 / 平均推理时间
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        
        return torch.tensor(fps, dtype=torch.float32)
    
    def compute(self):
        if self.mat is None or self.mat.sum() == 0:  # <<< 更健壮：防止空矩阵时报错
            n = self.num_classes
            zero = torch.tensor(0.0)
            acc_global = zero
            recall     = torch.zeros(n)
            precision  = torch.zeros(n)
            iou        = torch.zeros(n)
            dice       = torch.zeros(n)
            return acc_global, recall, precision, iou, dice

        h = self.mat.float()
        acc_global = torch.diag(h).sum() / (h.sum() + 1e-8)
        recall     = torch.diag(h) / (h.sum(1) + 1e-8)
        precision  = torch.diag(h) / (h.sum(0) + 1e-8)
        iou        = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h) + 1e-8)
        dice       = 2 * torch.diag(h) / (h.sum(1) + h.sum(0) + 1e-8)
        return acc_global, recall, precision, iou, dice

    def __str__(self):
        acc_global, recall, precision, iou, dice = self.compute()
        
        # 计算MAP和FPS
        ap_per_class, map_score = self.compute_map()
        fps = self.compute_fps()
        
        base = (
            'macc: {:.4f}\n'
            'mIoU: {:.4f}\n'
            'precision: {}\n'
            'recall: {}\n'
            'IoU: {}\n'
            'Dice: {}\n'
            'mDice: {:.4f}\n'
            'AP: {}\n'
            'mAP: {:.4f}\n'
            'FPS: {:.2f}\n'
        ).format(
            acc_global.item(),
            iou.mean().item(),
            ['{:.4f}'.format(i) for i in precision.tolist()],
            ['{:.4f}'.format(i) for i in recall.tolist()],
            ['{:.4f}'.format(i) for i in iou.tolist()],
            ['{:.4f}'.format(i) for i in dice.tolist()],
            dice.mean().item(),
            ['{:.4f}'.format(i) for i in ap_per_class.tolist()],
            map_score.item(),
            fps.item(),
        )

        # <<< 新增：把额外指标追加到打印内容
        if self._extra:
            for k, v in self._extra.items():
                base += f'{k}: {v}\n'
                
        return base
    def to_dict(self):
        """Convert metrics to dictionary by parsing the string representation.
        Avoids recomputing metrics by leveraging the existing __str__ output.
        """
        # Get the string representation
        str_repr = self.__str__()
        
        # Initialize the result dictionary
        result = {}
        
        # Process each line
        for line in str_repr.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Split key and value
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle list values
                if value.startswith('[') and value.endswith(']'):
                    # Remove brackets and split elements
                    items = value[1:-1].split(',')
                    value = [item.strip().strip("'\"") for item in items]
                
                result[key] = value
        
        return result