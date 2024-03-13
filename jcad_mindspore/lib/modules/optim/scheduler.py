import mindspore.nn as nn
import mindspore.ops as ops

# class PolyLr(nn.LearningRateSchedule):
#     def __init__(self, gamma, max_iteration, minimum_lr=0, warmup_iteration=0):
#         super(PolyLr, self).__init__()
#         self.gamma = gamma
#         self.max_iteration = max_iteration
#         self.minimum_lr = minimum_lr
#         self.warmup_iteration = warmup_iteration
#
#     def poly_lr(self, base_lr, step):
#         return (base_lr - self.minimum_lr) * ((1 - (step / self.max_iteration)) ** self.gamma) + self.minimum_lr
#
#     def warmup_lr(self, base_lr, alpha):
#         return base_lr * (1 / 10.0 * (1 - alpha) + alpha)
#
#     def construct(self, global_step):
#         if global_step < self.warmup_iteration:
#             alpha = ops.cast(global_step, mindspore.float32) / self.warmup_iteration
#             lr = self.warmup_lr(self.learning_rate, alpha)
#         else:
#             lr = self.poly_lr(self.learning_rate, global_step)
#         return lr