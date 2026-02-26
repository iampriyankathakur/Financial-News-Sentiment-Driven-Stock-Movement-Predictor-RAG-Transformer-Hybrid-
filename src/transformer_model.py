import torch.nn as nn

class StockTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
    def forward(self,x):
        return self.fc(x)
