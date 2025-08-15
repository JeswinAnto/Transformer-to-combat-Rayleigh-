import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEqualizer(nn.Module):
    def __init__(self, feat_dim, d_model=32, nhead=2, num_layers=1, dim_feedforward=32, output_dim=4):
        super(TransformerEqualizer, self).__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Linear layer to project input features -> d_model
        self.input_proj = nn.Linear(feat_dim, d_model)
        
        # Positional encoding (optional, can help with sequence modeling)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, feat_dim)
        batch_size, seq_len, feat_dim = x.shape
        
        # Project to model dimension
        x = self.input_proj(x)  # -> (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer(x)  # -> (batch, seq_len, d_model)
        
        # Project to output
        x = self.output_proj(x)  # -> (batch, seq_len, output_dim)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)