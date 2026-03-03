import torch
import torch.nn as nn


class CNNLSTMDiscriminator(nn.Module):
    """
    CNN + LSTM Hybrid model for deepfake audio detection.
    
    Architecture:
    - CNN: Extracts spatial features from spectrogram (local patterns, artifacts)
    - LSTM: Learns temporal patterns (how artifacts change over time)
    - Bidirectional LSTM: Processes sequence forwards and backwards
    
    Input shape: (batch_size, 1, 128, 128) - grayscale spectrogram
    Output shape: (batch_size, 1) - binary classification (real/fake)
    """
    
    def __init__(self, lstm_hidden_size=128, lstm_num_layers=2, dropout_rate=0.3):
        super(CNNLSTMDiscriminator, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout_rate = dropout_rate
        
        # ====== CNN Feature Extractor ======
        # Extracts spatial features from spectrogram
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x128 -> 64x64
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        # No pooling - keep 16x16 for LSTM temporal dimension
        
        self.dropout_cnn = nn.Dropout(dropout_rate)
        
        # ====== LSTM Temporal Sequence Processor ======
        # Input shape after CNN: (batch, 256, 16, 16)
        # Reshape to: (batch, 16 (height), 256*16 (flattened width)) 
        # LSTM will see: time_steps=16, input_size=4096
        
        cnn_output_height = 16
        cnn_output_width = 16
        cnn_output_channels = 256
        lstm_input_size = cnn_output_width * cnn_output_channels  # 16 * 256 = 4096
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Bidirectional LSTM outputs: lstm_hidden_size * 2
        lstm_output_size = lstm_hidden_size * 2
        
        self.dropout_lstm = nn.Dropout(dropout_rate)
        
        # ====== Fully Connected Classification Head ======
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through CNN+LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 128, 128)
        
        Returns:
            output: Classification logits of shape (batch_size, 1)
        """
        
        # ====== CNN Feature Extraction ======
        # x shape: (batch, 1, 128, 128)
        
        # Block 1
        x = self.conv1(x)                           # (batch, 32, 128, 128)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)                           # (batch, 32, 64, 64)
        x = self.dropout_cnn(x)
        
        # Block 2
        x = self.conv2(x)                           # (batch, 64, 64, 64)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)                           # (batch, 64, 32, 32)
        x = self.dropout_cnn(x)
        
        # Block 3
        x = self.conv3(x)                           # (batch, 128, 32, 32)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)                           # (batch, 128, 16, 16)
        x = self.dropout_cnn(x)
        
        # Block 4
        x = self.conv4(x)                           # (batch, 256, 16, 16)
        x = self.bn4(x)
        x = self.relu4(x)
        
        batch_size = x.size(0)
        
        # ====== Reshape for LSTM ======
        # x shape: (batch, 256, 16, 16)
        # Transpose to: (batch, 16, 256, 16)
        x = x.transpose(1, 2)                       # (batch, 16, 256, 16)
        
        # Flatten frequency dimension (height): (batch, 16, 256*16)
        x = x.contiguous().view(batch_size, 16, -1)  # (batch, 16, 4096)
        
        # ====== LSTM Processing ======
        # x shape: (batch, time_steps=16, features=4096)
        lstm_out, (h_n, c_n) = self.lstm(x)        # lstm_out: (batch, 16, 256)
        
        # Take output from last timestep
        # (bidirectional LSTM outputs last hidden state from both directions)
        last_output = lstm_out[:, -1, :]            # (batch, 256)
        
        x = self.dropout_lstm(last_output)
        
        # ====== Classification Head ======
        x = self.fc1(x)                             # (batch, 128)
        x = self.relu_fc(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)                             # (batch, 1)
        
        return x


def create_cnn_lstm_model(lstm_hidden_size=128, lstm_num_layers=2, dropout_rate=0.3):
    """
    Factory function to create CNN+LSTM model.
    
    Args:
        lstm_hidden_size: Hidden dimension for LSTM layers
        lstm_num_layers: Number of LSTM layers (stacked)
        dropout_rate: Dropout rate for regularization
    
    Returns:
        CNNLSTMDiscriminator model
    """
    model = CNNLSTMDiscriminator(
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        dropout_rate=dropout_rate
    )
    return model
