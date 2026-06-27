""" 
训练日志记录器（单 CSV，支持多 lead / 多步预测）

本项目的需求：
- 只生成 1 个 CSV（默认记录 *land* 评估口径，即 mask==0 的有效像素）
- 如果是多步预测（例如同时预测 1/2/4/8/16 个 8-day step 之后），
  CSV 里要同时保存这 5 种验证结果 + 这 5 种测试结果。

约定：
- train.py 里传入的 metrics 结构为：
    metrics_by_lead = {
        lead_time(int): {
            'loss': float,
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'auroc': float,
            'auprc': float,
        },
        ...
    }
- lead_time 的单位由数据集定义（当前通常是 "8-day step"）。
"""

import os
import csv
from datetime import datetime
from typing import Dict, List, Optional, Union


class TrainingLogger:
    """训练日志记录器"""

    BASE_METRICS = [
        'loss',
        'accuracy',
        'precision',
        'recall',
        'f1',
        'auroc',
        'auprc',
    ]

    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        lead_times: Optional[Union[int, List[int]]] = None,
        file_suffix: str = 'land',
    ):
        """
        Args:
            log_dir: 日志目录
            experiment_name: 实验名（不传则用时间戳）
            lead_times: int 或 list[int]，例如 1 或 [1,2,4,8,16]
            file_suffix: 文件后缀，默认 land
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        self.lead_times = self._normalize_lead_times(lead_times)

        # 单 CSV
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{file_suffix}.csv")

        # 动态列
        self.fieldnames = self._build_fieldnames(self.lead_times)
        self._init_csv()

        print("日志将保存到:")
        print(f"  - {self.log_file}")

    @staticmethod
    def _normalize_lead_times(lead_times: Optional[Union[int, List[int]]]) -> List[int]:
        if lead_times is None:
            return [1]
        if isinstance(lead_times, int):
            return [int(lead_times)]
        if isinstance(lead_times, (list, tuple)):
            if len(lead_times) == 0:
                return [1]
            return [int(x) for x in lead_times]
        raise TypeError(f"lead_times must be int or list[int], got: {type(lead_times)}")

    @classmethod
    def _build_fieldnames(cls, lead_times: List[int]) -> List[str]:
        # 基础信息
        fields: List[str] = [
            'epoch',
            'train_loss',
            'learning_rate',
            'timestamp',
        ]

        # 验证：每个 lead 一套
        for lt in lead_times:
            for m in cls.BASE_METRICS:
                fields.append(f"val_{m}_t{lt}")

        # 验证均值（方便排序/画图/选best）
        fields.extend([
            'val_auprc_mean',
            'val_auroc_mean',
        ])

        # 测试：每个 lead 一套
        for lt in lead_times:
            for m in cls.BASE_METRICS:
                fields.append(f"test_{m}_t{lt}")

        # 测试均值
        fields.extend([
            'test_auprc_mean',
            'test_auroc_mean',
        ])

        return fields

    def _init_csv(self):
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def _mean_metric(self, metrics_by_lead: Dict[int, Dict[str, float]], key: str) -> str:
        vals: List[float] = []
        for lt in self.lead_times:
            m = metrics_by_lead.get(lt)
            if m is None:
                continue
            if key in m and m[key] is not None:
                try:
                    vals.append(float(m[key]))
                except Exception:
                    pass
        if len(vals) == 0:
            return ''
        return f"{sum(vals) / len(vals):.6f}"

    def _fill_metrics(self, row: Dict[str, str], metrics_by_lead: Dict[int, Dict[str, float]], prefix: str):
        for lt in self.lead_times:
            m = metrics_by_lead.get(lt, {})
            for k in self.BASE_METRICS:
                col = f"{prefix}_{k}_t{lt}"
                if k in m and m[k] is not None:
                    try:
                        row[col] = f"{float(m[k]):.6f}"
                    except Exception:
                        row[col] = str(m[k])

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics_by_lead: Dict[int, Dict[str, float]],
        learning_rate: Optional[float] = None,
    ):
        """记录一个 epoch 的训练 + 验证（多 lead）。"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 先给所有字段置空，避免缺列
        row: Dict[str, str] = {k: '' for k in self.fieldnames}

        row['epoch'] = str(epoch)
        row['train_loss'] = f"{float(train_loss):.6f}"
        row['learning_rate'] = f"{float(learning_rate):.8f}" if learning_rate is not None else ''
        row['timestamp'] = timestamp

        # 填充验证
        self._fill_metrics(row, val_metrics_by_lead, prefix='val')
        row['val_auprc_mean'] = self._mean_metric(val_metrics_by_lead, 'auprc')
        row['val_auroc_mean'] = self._mean_metric(val_metrics_by_lead, 'auroc')

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def log_final_test(self, test_metrics_by_lead: Dict[int, Dict[str, float]], best_epoch: int):
        """记录最终测试（多 lead）。"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row: Dict[str, str] = {k: '' for k in self.fieldnames}
        row['epoch'] = f"FINAL (Best Epoch: {best_epoch})"
        row['timestamp'] = timestamp

        self._fill_metrics(row, test_metrics_by_lead, prefix='test')
        row['test_auprc_mean'] = self._mean_metric(test_metrics_by_lead, 'auprc')
        row['test_auroc_mean'] = self._mean_metric(test_metrics_by_lead, 'auroc')

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

        print("\n✓ 最终测试结果已记录到:")
        print(f"  - {self.log_file}")

    def get_log_path(self) -> str:
        return self.log_file


if __name__ == '__main__':
    # 简单自测
    logger = TrainingLogger(log_dir='./log', experiment_name='debug', lead_times=[1, 2, 4, 8, 16])
    dummy_val = {lt: {'loss': 0.1 * i, 'accuracy': 0.9, 'precision': 0.5, 'recall': 0.4, 'f1': 0.45, 'auroc': 0.8, 'auprc': 0.6 + 0.01 * i}
                 for i, lt in enumerate([1, 2, 4, 8, 16])}
    logger.log_epoch(epoch=1, train_loss=0.123, val_metrics_by_lead=dummy_val, learning_rate=1e-4)
    logger.log_final_test(test_metrics_by_lead=dummy_val, best_epoch=1)
    print("Log saved:", logger.get_log_path())
