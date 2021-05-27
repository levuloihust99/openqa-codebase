from argparse import Namespace
import yaml
import os
from typing import Dict


def write_config(config_path: str, config: Dict):
    dirname = os.path.dirname(config_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(config_path, "w") as writer:
        yaml.dump(config, writer)


def spread_samples_greedy(global_batch_size, num_replicas, base_replica_batch_size):
    if global_batch_size < num_replicas:
        return [global_batch_size]

    div = global_batch_size // base_replica_batch_size
    remain = global_batch_size % base_replica_batch_size
    spread = [base_replica_batch_size] * div + [remain]
    while True:
        if len(spread) == num_replicas:
            break
        
        base_replica_batch_size = base_replica_batch_size // 2
        idx = len(spread) - 1
        while True:
            if idx < 0:
                break

            value = spread[idx]
            if value > base_replica_batch_size:
                spread.pop(idx)
                spread.insert(idx, value - base_replica_batch_size)
                spread.insert(idx, base_replica_batch_size)

            idx = idx - 1
            if len(spread) == num_replicas:
                break

    return spread


def spread_samples_equally(global_batch_size, num_replicas, base_replica_batch_size):
    if global_batch_size < base_replica_batch_size:
        return [global_batch_size], -1, -1

    while global_batch_size < base_replica_batch_size * num_replicas:
        base_replica_batch_size = base_replica_batch_size // 2
        if base_replica_batch_size == 0:
            break

    remainder = global_batch_size - base_replica_batch_size * num_replicas
    return [base_replica_batch_size] * num_replicas, remainder, base_replica_batch_size