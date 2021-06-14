from argparse import Namespace
import yaml
import os
from typing import Dict
from transformers import BertModel, TFBertModel


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


def spread_samples_equally(global_batch_size, num_replicas, base_replica_batch_size, init_batch_size):
    if global_batch_size < init_batch_size:
        return [global_batch_size], -1, -1

    flag = False
    while global_batch_size < base_replica_batch_size * num_replicas:
        base_replica_batch_size = base_replica_batch_size // 2
        if base_replica_batch_size == 0:
            flag = True
            break
    
    if flag:
        return [global_batch_size], -1, -1

    remainder = global_batch_size - base_replica_batch_size * num_replicas
    return [base_replica_batch_size] * num_replicas, remainder, base_replica_batch_size


def check_compability(
    torch_model: BertModel,
    tf_model: TFBertModel
):
    torch_weights = []
    for k, v in torch_model.state_dict().items():
        if k == "embeddings.position_ids":
            print("im here")
            continue
        if not k.startswith("embeddings.") and k.endswith(".weight"):
            torch_weights.append(v.t().numpy())
        else:
            torch_weights.append(v.numpy())
    torch_weights[1], torch_weights[2] = torch_weights[2], torch_weights[1]

    tf_weights = tf_model.get_weights()

    check = [(torch_weight == tf_weight).all() for torch_weight, tf_weight in zip(torch_weights, tf_weights)]
    return all(check)