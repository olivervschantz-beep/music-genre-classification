import torch #
import os
import json

def add_one(number):
    """Add one to a number""" #Atte did this
    return number + 1


def load_json(file_path):
    """Load a json file"""
    with open(file_path, 'r') as f:
        return json.load(f)
    
def save_json(data, file_path):
    """Save data to a json file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_checkpoint(checkpoint_path, model, optimizer, iter):
    torch.save({
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iter = checkpoint['iter']
    return model, optimizer, iter
