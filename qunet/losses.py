import torch, torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, beta = 1.0):        
        super().__init__()    
        self.beta = beta

    def forward(self, pred, true, eps=1e-8):
        """ loss-function """
        return 1 - self.score(pred, true)

    def score(self, pred, true, eps=1e-8):
        """ 
        Sørensen–Dice score for all samples in batch.        
        For binary metric should be: pred = sigmoid(ligits) > 0.5
        For loss, do not forget: loss = 1 - dice_score()

        Args
            pred, true: (B,1,H,W) or (B, H*W) after sigmoid()                         
        """

        pred, true = pred.flatten().float(), true.flatten().float()
        assert len(pred) == len(true), f"pred.shape:{pred.shape},  true.shape:{true.shape}"
        
        np = true.sum()                      # number of positive in true
        tp = (pred *    true) .sum()         # true positive
        fp = (pred * (1-true)).sum()         # false positive               

        precis = tp / (tp + fp + eps)
        recall = tp / (np + eps)

        beta2 = self.beta**2
        dice   = (1+beta2) * (precis*recall) / (beta2*precis + recall + eps)

        return dice                          # (B,1)

#===============================================================================

class IoULoss(nn.Module):
    def __init__(self):        
        super().__init__()    

    def forward(self, pred, true, eps=1e-8):
        """ loss-function """
        return 1 - self.score(pred, true)

    def score(self, pred, true, eps=1e-8):
        """ 
        Intersection over Union       
        """
        pred, true = pred.flatten().float(), true.flatten().float()
        assert len(pred) == len(true), f"pred.shape:{pred.shape},  true.shape:{true.shape}"
                
        i = (pred *     true)               # intersection
        u = pred + true - i                 # union
        return i.sum() / (u.sum() + eps)

#===============================================================================