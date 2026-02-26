import torch
from transformer_model import StockTransformer

def train_model(X, y):
    model = StockTransformer(X.shape[1])
    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    return model
