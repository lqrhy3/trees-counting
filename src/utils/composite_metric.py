from itertools import chain
from typing import Dict, Tuple
from torchmetrics import Metric


class CompositeMetric:
    def __init__(self, density_metrics: Dict[str, Metric], tree_count_metrics: Dict[str, Metric]):
        self.density_metrics = density_metrics
        self.tree_count_metrics = tree_count_metrics

    def __call__(self, outputs, targets, pred_counts, tgt_counts) -> Tuple[Dict[str, float], Dict[str, float]]:
        density_metric_values = dict()
        for metric_name, metric_object in self.density_metrics.items():
            density_metric_values[metric_name] = metric_object(outputs, targets).item()

        tree_count_metric_values = dict()
        if tgt_counts > 10:
            for metric_name, metric_object in self.tree_count_metrics.items():
                tree_count_metric_values[metric_name] = metric_object(pred_counts, tgt_counts).item()

        return density_metric_values, tree_count_metric_values

    def reset(self):
        for metric_object in chain(self.density_metrics.values(), self.tree_count_metrics.values()):
            metric_object.reset()

    def compute(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        density_metric_values = dict()
        for metric_name, metric_object in self.density_metrics.items():
            density_metric_values[metric_name] = metric_object.compute().item()

        tree_count_metric_values = dict()
        for metric_name, metric_object in self.tree_count_metrics.items():
            tree_count_metric_values[metric_name] = metric_object.compute().item()

        return density_metric_values, tree_count_metric_values
