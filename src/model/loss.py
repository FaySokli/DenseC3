from torch import einsum, nn
from torch.nn.functional import relu
import torch

class MultipleRankingLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, device):
        super(MultipleRankingLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()
        self.BCELoss = nn.BCEWithLogitsLoss()
        
        self.device = device
    def forward(
        self,
        anchors_classes,
        anchors_pred,
        anchors,
        positives
    ):
        pw_similarity = torch.mm(anchors, positives.T)
        labels = torch.tensor([x for x in range(anchors.shape[0])], device=self.device)
        
        # import ipdb
        # ipdb.set_trace()
        cls_loss = self.BCELoss(anchors_pred, anchors_classes)
        pw_loss = self.CELoss(pw_similarity, labels)
        
        return pw_loss, cls_loss, .5 * pw_loss + .5 * cls_loss
    
    def val_forward(
        self,
        anchors_classes,
        anchors_pred,
        anchors,
        positives
    ):
        pw_similarity = torch.mm(anchors, positives.T)
        labels = torch.tensor([x for x in range(anchors.shape[0])], device=self.device)
        
        cls_loss = self.BCELoss(anchors_pred, anchors_classes)
        pw_loss = self.CELoss(pw_similarity, labels)
        
        return pw_loss, cls_loss, .5 * pw_loss + .5 * cls_loss, (pw_similarity.argmax(dim=1, keepdim=True).squeeze() == labels)


class MultipleRankingLossBiEncoder(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, device, temperature=1):
        super(MultipleRankingLossBiEncoder, self).__init__()
        self.CELoss = nn.CrossEntropyLoss()
        self.temperature = temperature
        
        self.device = device
    def no_acc_forward(
        self,
        anchors,
        positives
    ):
        pw_similarity = torch.mm(anchors / self.temperature, positives.T)
        labels = torch.tensor([x for x in range(anchors.shape[0])], device=self.device)
        
        pw_loss = self.CELoss(pw_similarity, labels)
        
        return pw_loss
    
    def forward(
        self,
        anchors,
        positives
    ):
        pw_similarity = torch.mm(anchors / self.temperature, positives.T)
        labels = torch.tensor([x for x in range(anchors.shape[0])], device=self.device)
        
        pw_loss = self.CELoss(pw_similarity, labels)
        
        return pw_loss, (pw_similarity.argmax(dim=1, keepdim=True).squeeze() == labels)

    def val_forward(self, anchors, positives):
        return self.forward(anchors, positives)



class TripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def no_acc_forward(self, anchors, positives):
        pos_scores = einsum('ij,ij->i', anchors, positives) 

        # only you can have a batch with more than 1 element do random negative batching:
        
        if positives.shape[0] > 1:
            repeated_pos_scores = pos_scores.view(-1,1).repeat_interleave(
                (positives.shape[0] - 1), dim=1
            )
            query_pos_scores = anchors @ positives.T # batch_size * batch_size
            
            masks = [[j for j in range(anchors.shape[0]) if j != i] for i in range(anchors.shape[0])]
            neg_docs_score = query_pos_scores.gather(1, torch.tensor(masks).to(query_pos_scores.device))

            
            in_batch_loss = relu(self.margin - repeated_pos_scores + neg_docs_score).mean()
        
        return in_batch_loss
    
    def val_forward(self, anchors, positives):
        return self.forward(anchors, positives)
    
    def forward(self, anchors, positives):
        pos_scores = einsum('ij,ij->i', anchors, positives) 

        # only you can have a batch with more than 1 element do random negative batching:
        
        if positives.shape[0] > 1:
            repeated_pos_scores = pos_scores.view(-1,1).repeat_interleave(
                (positives.shape[0] - 1), dim=1
            )
            query_pos_scores = anchors @ positives.T # batch_size * batch_size
            
            masks = [[j for j in range(anchors.shape[0]) if j != i] for i in range(anchors.shape[0])]
            neg_docs_score = query_pos_scores.gather(1, torch.tensor(masks).to(query_pos_scores.device))

            
            in_batch_loss = relu(self.margin - repeated_pos_scores + neg_docs_score).mean()
            
            accuracy = repeated_pos_scores.view(-1) > neg_docs_score.view(-1)
            
            return in_batch_loss, accuracy
        
        in_batch_loss = relu(self.margin - pos_scores + pos_scores).mean()
        return in_batch_loss, pos_scores > 0
        
