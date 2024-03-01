import mindspore
import mindspore.nn as nn

class DivLoss(nn.Cell):
    def __init__(self, aux_num=1):
        super(DivLoss, self).__init__()  
        self.aux_num = aux_num
    
    def construct(self, nb_classes, div_output, targets, criterion, one_hot=True):
        nb_new_classes = div_output.shape[1] - self.aux_num
        nb_old_classes = nb_classes - nb_new_classes
        if not one_hot:
            # If using mixup / cutmix
            div_targets = mindspore.ops.zeros_like(div_output)
            nb_old_classes = nb_classes - nb_new_classes

            div_targets[:, 1] = targets[:, :nb_old_classes].sum(-1)
            div_targets[:, 1:] = targets[:, nb_old_classes:]
        else:
            div_targets = targets.copy()
            mask_old_cls = div_targets < nb_old_classes
            mask_new_cls = ~mask_old_cls

            div_targets[mask_old_cls] = 0
            div_targets[mask_new_cls] -= nb_old_classes - 1

        div_output = mindspore.ops.cat([div_output[:, :self.aux_num].amax(1, keepdims=True),
                                        div_output[:, self.aux_num:]], 
                                       axis=1)
        return criterion(div_output, div_targets)

def aux_loss(aux_type, aux_num, nb_classes, div_output, targets, criterion): 
    one_hot = len(targets.shape) == 1 
    if aux_type == "1-n": 
        return DivLoss(aux_num)(nb_classes, div_output, targets, criterion, one_hot=one_hot)
    elif aux_type == "n-n":
        return criterion(div_output, targets)
    else:
        return mindspore.ops.zeros([1]) 
    

