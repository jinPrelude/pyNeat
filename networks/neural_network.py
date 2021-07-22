from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstracts import BaseNetwork


class GymEnvModel(BaseNetwork):
    def __init__(self, num_state=8, num_action=4, discrete_action=True, gru=True):
        super(GymEnvModel, self).__init__()
        self.num_action = num_action
        self.fc1 = nn.Linear(num_state, 32)
        self.use_gru = gru
        if self.use_gru:
            self.gru = nn.GRU(32, 32)
            self.h = torch.zeros([1, 1, 32], dtype=torch.float)
        self.fc2 = nn.Linear(32, num_action)
        self.discrete_action = discrete_action

        self.model_shape = []
        for param in self.parameters():
            shape = param.data.numpy().shape
            size = deepcopy(param.data.numpy()).flatten().size
            self.model_shape.append((shape, size))

    def forward(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            x = x.unsqueeze(0)
            x = torch.tanh(self.fc1(x))
            if self.use_gru:
                x, self.h = self.gru(x, self.h)
                x = torch.tanh(x)
            x = self.fc2(x)
            if self.discrete_action:
                x = F.softmax(x.squeeze(), dim=0)
                x = torch.argmax(x)
            else:
                x = torch.tanh(x.squeeze())
            x = x.detach().cpu().numpy()

            return x

    def reset(self):
        if self.use_gru:
            self.h = torch.zeros([1, 1, 32], dtype=torch.float)

    def zero_init(self):
        for param in self.parameters():
            param.data = torch.zeros(param.shape)

    def get_param_list(self):
        param_list = []
        for param in self.parameters():
            param_list.append(param.data.numpy())
        return self._flatten_param(param_list)

    def apply_param(self, param_lst: list):
        param_lst = self._restore_param(param_lst)
        count = 0
        for p in self.parameters():
            p.data = torch.tensor(param_lst[count]).float()
            count += 1

    @staticmethod
    def _flatten_param(param_list):
        flattened_param = []
        for param in param_list:
            flattened_param.append(deepcopy(param).flatten())
        flattened_param = np.concatenate(flattened_param)
        return flattened_param

    def _restore_param(self, param_array):
        idx = 0
        restored_param = []
        for (shape, size) in self.model_shape:
            param = param_array[idx : idx + size]
            param = np.reshape(param, shape)
            restored_param.append(param)
            idx += size

        return restored_param
