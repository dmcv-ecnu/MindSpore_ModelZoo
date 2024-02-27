from collections import defaultdict
import numpy as np
import numbers
import math


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MultiAverageMeter():
    def __init__(self):
        self.values = defaultdict(AverageMeter)
    
    def update(self, name, value):
        self.values[name].update(value)

    @property
    def avg_all(self):
        m = 0
        for v in self.values.values():
            m += v.avg
        if len(self.values) == 0:
            return 0
        return m / len(self.values) 
    
    @property
    def avg_per(self):
        m = {}
        for k, v in self.values.items():
            m[k] = v.avg
        return m
    
    def get(self, name):
        return self.values[name].avg
    

def accuracy(output, target, topk=(1,)):
    maxk = min(max(topk), output.shape[1])  # 进行到此处  Soap
    batch_size = target.shape[0]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [(correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size).item() for k in topk]
