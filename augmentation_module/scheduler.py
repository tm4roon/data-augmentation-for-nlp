# -*- coding: utf-8 -*-


from abc import abstractmethod


# def get_scheduler(method):
#     if method == 'constant':
#         return ConstantAR
#     elif method == 'linear':
#         return LinearAR
#     elif method == 'exponential':
#         return ExponentialAR


class ARscheduler(object):
    def __init__(self, iterator, augmentation_rate):
        self.iterator = iterator
        self.augmentation_rate = augmentation_rate

    @abstractmethod
    def initialize(self):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError


class ConstantAR(ARscheduler):
    def __init__(self, iterator, augmentation_rate):
        super(ConstantAR, self).__init__(iterator, augmentation_rate)
        self.initialize()

    def initialize(self):
        self.iterator.augmentation_rate = self.augmentation_rate

    def step(self):
        self.iterator.augmentation_rate = self.augmentation_rate


class LinearAR(ARscheduler):
    def __init__(self, iterator, augmentation_rate, max_epoch):
        super(LinearAR, self).__init__(iterator, augmentation_rate)
        self.decay = - self.augmentation_rate / max_epoch
        self.initialize()

    def initialize(self):
        self.iterator.augmentation_rate = self.augmentation_rate

    def step(self):
        rate = self.iterator.augmentation_rate + self.decay
        self.iterator.augmentation_rate = max(0.0, rate)


class ExponentialAR(ARscheduler):
    def __init__(self, iterator, augmentation_rate):
        super(ExponentialAR, self).__init__(iterator, augmentation_rate)
        self.initialize()

    def initialize(self):
        self.n_steps = 0
        self.iterator.augmentation_rate = self.augmentation_rate ** (self.n_steps + 1)

    def step(self):
        self.n_steps += 1
        self.iterator.augmentation_rate = self.augmentation_rate ** (self.n_steps + 1)


class StepAR(ARscheduler):
    def __init__(self, iterator, augmentation_rate, step_size, decay):
        super(StepAR, self).__init__(iterator, augmentation_rate)
        self.step_size = step_size
        self.decay = decay
        self.initialize()

    def initialize(self):
        self.n_steps = 0
        self.iterator.augmentation_rate = self.augmentation_rate

    def step(self):
        self.n_steps += 1
        if (self.n_steps % self.step_size) == 0:
            self.iterator.augmentation_rate = self.iterator.augmentation_rate * self.decay
        

class WarmupConstantAR(ARscheduler):
    def __init__(self, iterator, augmentation_rate, warmup_epoch, total_epoch):
        super(WarmupConstantAR, self).__init__(iterator, augmentation_rate)
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch
        self.initialize()

    def initialize(self):
        self.n_steps = 0
        self.iterator.augmentation_rate = 0.0

    def step(self):
        self.n_steps += 1
        if self.n_steps < self.warmup_epoch:
            rate = self.n_steps * self.augmentation_rate / self.warmup_epoch
        else:
            rate = self.augmentation_rate
        self.iterator.augmentation_rate = rate


class WarmupLinearAR(ARscheduler):
    def __init__(self, iterator, augmentation_rate, warmup_epoch, total_epoch):
        super(WarmupLinearAR, self).__init__(iterator, augmentation_rate)
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch
        self.initialize()

    def initialize(self):
        self.n_steps = 0
        self.iterator.augmentation_rate = 0.0

    def step(self):
        self.n_steps += 1
        if self.n_steps < self.warmup_epoch:
            rate = self.n_steps * self.augmentation_rate / self.warmup_epoch
        else:
            grad = self.augmentation_rate / (self.total_epoch - self.warmup_epoch)
            rate = max(0.0, -grad * self.n_steps + grad * self.total_epoch)
        self.iterator.augmentation_rate = rate

