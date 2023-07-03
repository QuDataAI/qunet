import torch, torch.nn as nn
import torch.nn.functional as F

#===============================================================================

class BCELoss(nn.Module):
    def __init__(self, pos = 1):        
        """
        pos - weight of positive class
        """
        super().__init__()            
        assert pos > 0, f"weight of positive class should be positive, but got {pos}"
        self.pos = pos

    def forward(self, pred, true, eps=1e-6):
        """ loss-function eps <= 1e-7 !!! """
        pred = torch.clamp(pred, min=eps, max=1-eps) 
        pred = pred.flatten()                          # for segmentation
        true = true.flatten() 
        loss = -( self.pos * true * torch.log(pred) + (1-true) * torch.log(1-pred) ).mean()
        return loss / (1+self.pos)

#===============================================================================

class DiceLoss(nn.Module):
    def __init__(self, beta = 1.0):        
        super().__init__()    
        self.beta = beta

    def forward(self, pred, true, eps=1e-6):
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

class BatchTtripletLoss(nn.Module):
    """Uses all valid triplets to compute Triplet loss

    Args:
        margin: Margin value in the Triplet Loss equation

    https://towardsdatascience.com/triplet-loss-advanced-intro-49a07b7d8905
    """
    def __init__(self, margin=1., pack_size = None):
        """Triplet loss (Ancor, Poitive, Negative):
        
        Args
        ------------
            margin    (float=1.):
                гиперпараметр не дающий классам
            pack_size (int = None):
                split the batch into packs of `pack_size` size to reduce the complexity of `pack*(B/pack)**3`
        """
        super().__init__()
        self.margin = margin
        self.pack   = pack_size
        self.triplets = 0
    
    #---------------------------------------------------------------------------

    def forward(self, embeddings, labels, eps=1e-8):
        """Computes loss value.

        Args:
        ------------
            embeddings: Batch of embeddings, e.g., output of the encoder. shape: (B,E)
            labels: Batch of integer labels associated with embeddings.   shape: (B,)

        Returns:
        ------------
            Scalar loss value.
        """
        if self.pack:
            loss, tot = 0, 0
            for i in range(0, len(embeddings),  self.pack):
                l, n = self.calc(embeddings[i: i+self.pack], 
                                 labels    [i: i+self.pack], eps=1e-8)
                loss = loss + l*n
                tot  = tot  + n
            self.triplets = tot
            return loss / tot
        else:
            loss, self.triplets = self.calc( embeddings, labels, eps=1e-8)
            return loss
    
    #---------------------------------------------------------------------------

    def calc(self, embeddings, labels, eps=1e-8):
        """
        Computes loss value. see forward
        """
        # step 1 - get distance matrix        
        distance_matrix = self.euclidean_distance_matrix(embeddings)     # (B,B)

        # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix        
        anchor_pos_dists = distance_matrix.unsqueeze(2)                  # (B,B,1)        
        anchor_neg_dists = distance_matrix.unsqueeze(1)                  # (B,1,B)
        # get loss values for all possible n^3 triplets        
        triplet_loss = anchor_pos_dists - anchor_neg_dists + self.margin # (B,B,B)

        # step 3 - filter out invalid or easy triplets by setting their loss values to 0        
        mask = self.get_triplet_mask(labels)                             # (B,B,B)        
        triplet_loss = triplet_loss * mask
        # easy triplets have negative loss values
        triplet_loss = F.relu(triplet_loss)

        # step 4 - compute scalar loss value by averaging positive losses
        num_positive_losses = (triplet_loss > eps).float().sum()
        triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)

        #print(mask.sum())
        return triplet_loss, mask.sum()

    #---------------------------------------------------------------------------

    def get_triplet_mask(self, labels):
        """compute a mask for valid triplets

        Args:
        ------------
            labels: Batch of integer labels. shape: (batch_size,)

        Returns:
        ------------
            Mask tensor to indicate which triplets are actually valid. Shape: (B,B,B)
            A triplet is valid if:
            `labels[i] == labels[j] and labels[i] != labels[k]` and `i`, `j`, `k` are different.
        """
        # step 1 - get a mask for distinct indices
        indices_eq = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device) #(B,B)
        indices_not_eq = torch.logical_not(indices_eq)    
        i_not_eq_j = indices_not_eq.unsqueeze(2)  # (B,B,1)        
        i_not_eq_k = indices_not_eq.unsqueeze(1)  # (B,1,B)        
        j_not_eq_k = indices_not_eq.unsqueeze(0)  # (1,B,B)
        # (B,B,B):
        distinct_indices = torch.logical_and(torch.logical_and(i_not_eq_j, i_not_eq_k), j_not_eq_k)

        # step 2 - get a mask for valid anchor-positive-negative triplets        
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B,B)        
        i_eq_j = labels_eq.unsqueeze(2)                          # (B,B,1)        
        i_eq_k = labels_eq.unsqueeze(1)                          # (B,1,B)        
        valid_indices = torch.logical_and(i_eq_j, torch.logical_not(i_eq_k)) # (B,B,B)

        # step 3 - combine two masks        
        return torch.logical_and(distinct_indices, valid_indices)
    
    #---------------------------------------------------------------------------

    def euclidean_distance_matrix(self, x, eps = 1e-8):
        """Efficient computation of Euclidean distance matrix

        Args:
        ------------
            x: Input tensor of shape (B,E)
    
        Returns:
        ------------
            Distance matrix of shape (B,B)
        """
        # step 1 - compute the dot product        
        dot_product = torch.mm(x, x.t())    # (B,B)

        # step 2 - extract the squared Euclidean norm from the diagonal        
        squared_norm = torch.diag(dot_product)  # (B,)

        # step 3 - compute squared Euclidean distances   (B,B)        
        distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)

        # get rid of negative distances due to numerical instabilities
        distance_matrix = F.relu(distance_matrix)

        # step 4 - compute the non-squared distances  
        # handle numerical stability derivative of the square root operation applied 
        # to 0 is infinite we need to handle by setting any 0 to eps
        mask = (distance_matrix == 0.0).float()

        # use this mask to set indices with a value of 0 to eps
        distance_matrix = distance_matrix +  mask * eps

        # now it is safe to get the square root
        distance_matrix = torch.sqrt(distance_matrix)

        # undo the trick for numerical stability
        distance_matrix = distance_matrix * (1.0 - mask)

        return distance_matrix    
    
    #---------------------------------------------------------------------------
    @staticmethod
    def unit_test():
        loss = BatchTtripletLoss(1., 0)
        B, E, C = 100, 128, 10
        embs, lbls = torch.randn(B, E), torch.randint(0,C, size=(B,))
        L = loss(embs, lbls)
        print(L)
        print(loss.triplets)


if __name__ == '__main__':    
    BatchTtripletLoss.unit_test()