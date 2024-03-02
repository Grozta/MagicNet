import os 
import torch

def load_checkpoint(pretrain_model_path,model,optimizer):
    if os.path.exists(pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path)
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['state_dict']

        # filter out unnecessary keys
        # pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()
        #                     if k.replace('module.', '') in model_dict}

        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # load the new state dict
        model.load_state_dict(model_dict)
        optimizer.update(checkpoint['optimizer_dict'])

        return checkpoint['lr'],checkpoint['iters']

def save_checkpoint(model, lr, optimizer, iters, save_weight_path):
    torch.save({
        'lr': lr,
        'iters': iters,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict()},
        save_weight_path)