import numpy as np
import utils.sharing_params as params
import torch
import torch.nn.functional as Func
import torch.nn as nn

def loss_compute(logits, labels, type='test'):
    # batch_size, num_classes = logits.shape
    # try:
    if type=='test':
        loss = Func.cross_entropy(logits, labels)
    else:
        pred=nn.functional.softmax(logits, 1)[:,1]
        loss=Func.smooth_l1_loss(pred,labels)
    # except:
    #     loss =Func.cross_entropy(logits, labels.unsqueeze(0))
    return loss


