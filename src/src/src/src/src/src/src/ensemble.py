import torch
import numpy as np

def ensemble_predict(gpt_pred, cnn_pred, alpha=0.6):
    """
    Weighted ensemble: alpha * GPT + (1-alpha) * CNN
    """
    return gpt_pred if alpha >= 0.5 else cnn_pred
