# src/Moksha_1/models/deep_quant/model.py
import torch
import torch.nn as nn

class FinancialNN3(nn.Module):
    """
    Standard Feedforward Network (NN3) from Gu, Kelly & Xiu (2020).
    Structure: Input -> 32 -> 16 -> 8 -> Output
    """
    def __init__(self, input_dim: int, dropout_rate: float = 0.5):
        super(FinancialNN3, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32), # Stabilizes learning
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Prevents overfitting (Crucial for finance)
            
            # Layer 2
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Layer 3
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            
            # Output Layer (Linear prediction of return)
            nn.Linear(8, 1)
        )
        
        # Initialize weights (He Initialization is best for ReLU)
        self._init_weights()

    def _init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)