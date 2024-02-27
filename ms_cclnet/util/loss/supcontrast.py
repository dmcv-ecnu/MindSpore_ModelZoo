"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

from mindspore import ops, nn

class SupConLoss(nn.Cell):
    def __init__(self):
        super(SupConLoss, self).__init__()
        self.temperature = 1.0
    def forward(self, text_features, image_features, t_label, i_targets): 
        batch_size = text_features.shape[0] 
        batch_size_N = image_features.shape[0] 
        mask = ops.equal(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float()

        logits = ops.div(ops.matmul(text_features, image_features.T),self.temperature)
        # for numerical stability
        logits_max, _ = ops.max(logits, axis=1, keepdims=True)
        logits = logits - logits_max.detach() 
        exp_logits = ops.exp(logits)
        log_prob = logits - ops.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 
        loss = - mean_log_prob_pos.mean()

        return loss