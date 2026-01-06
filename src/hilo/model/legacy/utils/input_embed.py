import torch.nn as nn

# import torch.nn.init as init


class InputEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        # # Initialize the weights using Xavier initialization
        # init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        embedded_x = self.embedding(x)
        return embedded_x
