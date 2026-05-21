# Note

## RmsNorm
token(x)输入：(batch_size, seq_len, hidden_dim)，故mean维度为-1，即hidden_dim维度，keepdim=True保持维度不变，输出(batch_size, seq_len, 1),而不是变成unsqueeze()了，再乘以权重weight，输出(batch_size, seq_len, hidden_dim)