from torch import nn


class OODDector(nn.Module):
    def __init__(self, num_classes=10, multi_class=True):
        super().__init__()
        self.fc_out = nn.Linear(8192, 2048)
        self.relu = nn.ReLU()
        if multi_class:
            k = num_classes + 1
        else:
            k = 1
        self.out_layer = nn.Linear(2048, k)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc_out(x)
        x = self.relu(x)
        x = self.out_layer(x)
        return x