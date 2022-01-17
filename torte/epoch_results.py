from dataclasses import dataclass


@dataclass
class EpochResults:
    loss: float  # epoch avg loss
    metrics: dict  # dictionary of metric_label -> metric value pairs
