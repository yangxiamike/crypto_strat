from torch import nn


class BaseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(BaseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, edge_index=None, edge_weight=None):
        """x expected shape [num_nodes_total, num_features, seq_len].
        We ignore graph edges in this baseline and run per-node sequence modeling.
        """
        # x: [batch size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x) # lstm_out : [batch size, seq_len, hidden_size]
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output