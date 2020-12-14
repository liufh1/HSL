import torch
import torch.nn as nn
from utils import *


def dot_sim(im, s):
    """
    Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, sims):
        target = torch.arange(sims.size(0)).long()
        if torch.cuda.is_available():
            target = target.cuda()

        return self.loss(sims, target)