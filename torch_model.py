import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.softmax(self.linear(x))