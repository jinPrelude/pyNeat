import numpy as np
from copy import deepcopy

# Reference:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py


class Optimizer(object):
    def __init__(self, pi, epsilon=1e-08):
        self.pi = pi
        self.epsilon = epsilon
        self.t = 0

    def update(self, delta_param_list):
        self.t += 1
        step = self._compute_step(delta_param_list)
        theta = self.pi.get_param_list()
        theta += step
        self.pi.apply_param(theta)

    def _compute_step(self, globalg):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.zero_param_list = deepcopy(self.pi).get_param_list() * 0
        self.m = deepcopy(self.zero_param_list)
        self.v = deepcopy(self.zero_param_list)

    def _compute_step(self, param_list):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * param_list
        self.v = self.beta2 * self.v + (1 - self.beta2) * (param_list * param_list)
        step = deepcopy(self.zero_param_list)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
