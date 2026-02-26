import torch.nn as nn

class StockTransformer(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=1,
            batch_first=True
        )
        self.fc = nn.Linear(input_dim,2)

    def forward(self,x):
        x = x.unsqueeze(1)
        attn_output,_ = self.attention(x,x,x)
        return self.fc(attn_output.squeeze(1))


attn_output, attn_weights = model.attention(x,x,x)

import matplotlib.pyplot as plt
plt.imshow(attn_weights.detach().numpy())
plt.colorbar()
