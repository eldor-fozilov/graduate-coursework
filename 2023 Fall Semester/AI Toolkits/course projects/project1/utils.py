import random
import torch
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def score(accu):
    return 100*sigmoid(0.1*(accu-75)) + 100*(1-sigmoid(2.5))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model

def reset(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
