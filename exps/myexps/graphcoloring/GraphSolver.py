import torch.nn as nn


import satnet


class GraphColoringSolver(nn.Module):
    def __init__(self, v_num, aux, m):
        super(GraphColoringSolver, self).__init__()
        n = (v_num ** 2) * (v_num + 1)
        self.sat = satnet.SATNet(n, m, aux)

    def forward(self, y_in, mask):
        out = self.sat(y_in, mask)
        return out