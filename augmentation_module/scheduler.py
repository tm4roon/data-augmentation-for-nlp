# -*- coding: utf-8 -*-


from abc import abstractmethod


def get_scheduler(method):
    if method == 'constant':
        return ConstantAR
    elif method == 'linear':
        return LinearAR
    elif method == 'exponential':
        return ExponentialAR


class ARscheduler(object):
    def __init__(self, init_rate, max_epoch):
        self.init_rate = init_rate
        self.max_epoch = max_epoch

    @abstractmethod
    def __call__(self, step):
        raise NotImplementedError


class ConstantAR(ARscheduler):
    def __call__(self, step):
        return self.init_rate


class LinearAR(ARscheduler):
    def __call__(self, epoch):
        current_rate = self.init_rate - self.init_rate * (epoch - 1) / self.max_epoch
        return max(0.0, current_rate)


# [TODO]
class ExponentialAR(ARscheduler):
    def __call__(self, step):
        raise NotImplementedError
