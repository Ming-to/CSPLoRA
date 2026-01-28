import torch
import torch.nn as nn
from typing import Dict


def format_params(num: int) -> str:
    if num >= 1000:
        return f"{num/1000:.2f}K"
    return str(num)


def count_parameters(model: nn.Module, verbose: bool = False) -> Dict[str, float]:
    def is_classifier_layer(name: str) -> bool:
        classifier_keywords = ['classifier', 'fc', 'linear', 'head']
        return any(keyword in name.lower() for keyword in classifier_keywords)

    total_params = 0
    classifier_params = 0
    non_classifier_params = 0

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_count = parameter.numel()
            total_params += param_count

            if is_classifier_layer(name):
                classifier_params += param_count
            else:
                non_classifier_params += param_count

            if verbose:
                print(f"{name}: {format_params(param_count)} parameters "
                      f"{'(Classifier)' if is_classifier_layer(name) else '(Non-classifier)'}")

    results = {
        'total_trainable_params': total_params / 1000,
        'classifier_params': classifier_params / 1000,
        'non_classifier_params': non_classifier_params / 1000,
    }

    print(f"\nTotal trainable parameters: {format_params(total_params)}")

    return results
