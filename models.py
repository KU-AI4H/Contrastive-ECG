import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer



class ECG_Classifier(nn.Module):
    def __init__(self, encoder, embedded_size=256, num_classes=1, dropout=0.1):
        super(ECG_Classifier, self).__init__()
        self.encoder = encoder  # Use pretrained encoder
        
        # Two-layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedded_size, embedded_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedded_size, num_classes),
            nn.Sigmoid()  # Sigmoid for binary classification
        )

    def forward(self, x):
        x, _ = self.encoder(x)  # Get encoded features
        out = self.classifier(x)  # Classifier prediction
        return out

#############################################

class ECG_CNN_Encoder(nn.Module):
    def __init__(self, signal_length=5000, embedded_size=256, CL_embedded_size=64, kernel_size=15, dropout=0.1, alpha=0.1, seed=42):
        super(ECG_CNN_Encoder, self).__init__()

        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        self.signal_length = signal_length
        padding_size = int((kernel_size - 1) / 2)
            
        self.conv1 = nn.Conv1d(12, 32, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn1 = nn.BatchNorm1d(32)
        self.avgpool1 = nn.AvgPool1d(kernel_size=kernel_size, stride=2)
        
        self.conv2 = nn.Conv1d(32, 32, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn2 = nn.BatchNorm1d(32)
        self.avgpool2 = nn.AvgPool1d(kernel_size=kernel_size, stride=2) 
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn5 = nn.BatchNorm1d(64)

        self.flatten_size = self._get_flatten_size()
        self.fc1 = nn.Linear(self.flatten_size, 256)  

        self.alpha = alpha
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout_fc1 = nn.Dropout(dropout*2)

        # Contrastive learning projector
        self.contrastive_projector = nn.Sequential(
            nn.Linear(embedded_size, embedded_size),
            nn.ReLU(),
            nn.Linear(embedded_size, CL_embedded_size)
        )

    def _get_flatten_size(self):
        """Get the size of the flattened feature map after convolutions."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 12, self.signal_length)
            x = F.leaky_relu(self.conv1(dummy_input))
            x = F.leaky_relu(self.conv2(x))
            x = self.avgpool1(x)  
            x = F.leaky_relu(self.conv3(x))
            x = self.avgpool2(x)  
            x = F.leaky_relu(self.conv4(x))
            x = F.leaky_relu(self.conv5(x))
            return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=self.alpha))
        x = self.dropout2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=self.alpha))
        x = self.avgpool1(x)
        x = self.dropout3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=self.alpha))
        x = self.avgpool2(x)
        x = self.dropout4(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=self.alpha))
        x = self.dropout5(F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=self.alpha))
        
        x = torch.flatten(x, 1)
        x = self.dropout_fc1(F.leaky_relu(self.fc1(x), negative_slope=self.alpha))

        transfered_embd = self.contrastive_projector(x)  # Contrastive learning embedding
        
        return x, transfered_embd  # Encoder output & contrastive embedding



############################################################


class ECG_CNN_LSTM_Encoder(nn.Module):
    def __init__(self, signal_length=5000, embedded_size=256, CL_embedded_size=128, kernel_size=15, dropout=0.1, alpha=0.1, lstm_layers=2, seed=42):
        super(ECG_CNN_LSTM_Encoder, self).__init__()

        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.signal_length = signal_length
        padding_size = (kernel_size - 1) // 2

        lstm_hidden_size = embedded_size
            
        # CNN Layers
        self.conv1 = nn.Conv1d(12, 32, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn2 = nn.BatchNorm1d(32)
        self.avgpool2 = nn.AvgPool1d(kernel_size=kernel_size, stride=2)
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn3 = nn.BatchNorm1d(64)
        self.avgpool3 = nn.AvgPool1d(kernel_size=kernel_size, stride=2)
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout)

        self.conv5 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(dropout)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=64,  # CNN output channels
            hidden_size=lstm_hidden_size // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout_lstm = nn.Dropout(dropout)

        # Fully Connected Layer
        self.fc1 = nn.Linear(lstm_hidden_size, embedded_size)  # BiLSTM doubles hidden size
        self.dropout_fc1 = nn.Dropout(dropout*2)

        # Contrastive Learning Projector
        self.contrastive_projector = nn.Sequential(
            nn.Linear(embedded_size, embedded_size),
            nn.ReLU(),
            nn.Linear(embedded_size, CL_embedded_size)
        )

    def forward(self, x):
        # CNN Feature Extraction
        x = self.dropout1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1))

        x = self.dropout2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1))
        x = self.avgpool2(x)

        x = self.dropout3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1))
        x = self.avgpool3(x)

        x = self.dropout4(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1))

        x = self.dropout5(F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.1))

        # Prepare for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # Change from (batch, channels, seq_len) → (batch, seq_len, channels)

        # LSTM Feature Extraction
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last hidden state
        x = self.dropout_lstm(x) # Apply dropout

        # Fully Connected Layer
        x = self.dropout_fc1(F.leaky_relu(self.fc1(x), negative_slope=0.1))

        # Contrastive Learning Embedding
        transfered_embd = self.contrastive_projector(x)

        return x,transfered_embd   # Encoder output & contrastive embedding


############################################################

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
            
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_dim)
        # lstm output: (batch_size, seq_len, hidden_dim)
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)

        energy = torch.tanh(
            self.attention(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch_size, seq_len, hidden_dim)
        attention_scores = self.v(energy).squeeze(2)  # (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(
            1)  # (batch_size, input_dim)
        return context_vector, attention_weights


class ECG_CNN_LSTM_Encoder(nn.Module):
    def __init__(self, signal_length=5000, embedded_size=256, CL_embedded_size=128, kernel_size=15, dropout=0.1, alpha=0.1, lstm_layers=2, seed=42):
        super(ECG_CNN_LSTM_Encoder, self).__init__()

        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.signal_length = signal_length
        padding_size = (kernel_size - 1) // 2

        lstm_hidden_size = embedded_size
        
        # CNN Layers
        self.conv1 = nn.Conv1d(12, 32, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(32, 32, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn2 = nn.BatchNorm1d(32)
        self.avgpool2 = nn.AvgPool1d(kernel_size=kernel_size, stride=2)
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn3 = nn.BatchNorm1d(64)
        self.avgpool3 = nn.AvgPool1d(kernel_size=kernel_size, stride=2)
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout)

        self.conv5 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(dropout)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=64,  # CNN output channels
            hidden_size=lstm_hidden_size // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout_lstm = nn.Dropout(dropout)

        # Attention Layer
        self.attention = Attention(dim=lstm_hidden_size)

        # Fully Connected Layer
        self.fc1 = nn.Linear(lstm_hidden_size, embedded_size)  # BiLSTM doubles hidden size
        self.dropout_fc1 = nn.Dropout(dropout*2)

        # Contrastive Learning Projector
        self.contrastive_projector = nn.Sequential(
            nn.Linear(embedded_size, embedded_size),
            nn.ReLU(),
            nn.Linear(embedded_size, CL_embedded_size)
        )

    def forward(self, x):
        # CNN Feature Extraction
        x = self.dropout1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1))
        x = self.dropout2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1))
        x = self.avgpool2(x)

        x = self.dropout3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.1))
        x = self.avgpool3(x)

        x = self.dropout4(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.1))
        x = self.dropout5(F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.1))

        # Prepare for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # Change from (batch, channels, seq_len) → (batch, seq_len, channels)

        # LSTM Feature Extraction
        lstm_out, (hn, _) = self.lstm(x)
        context_vector, attention_weights = self.attention(lstm_out[:, -1, :],
                                                           lstm_out)  # Use last hidden state of LSTM

        # Fully Connected Layer
        x = self.dropout_fc1(F.leaky_relu(self.fc1(context_vector), negative_slope=0.1))

        # Contrastive Learning Embedding
        transfered_embd = self.contrastive_projector(x)

        return x, transfered_embd, attention_weights   # Encoder output, contrastive embedding, attention weights


####################################################################



class ResidualBlock(nn.Module):
    """Residual Block with 1D convolution."""
    def __init__(self, in_channels, out_channels, kernel_size=15, dropout=0.1):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        x += residual  # Residual connection
        x = self.relu(x)
        return x


class ECG_CNN_LSTM_Residual(nn.Module):
    def __init__(self, signal_length=5000, embedded_size=256, CL_embedded_size=128, kernel_size=15, dropout=0.1, lstm_layers=2, seed=42):
        super(ECG_CNN_LSTM_Residual, self).__init__()

        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.signal_length = signal_length
        lstm_hidden_size = embedded_size  # Ensure consistency in dimensions

        # CNN Blocks with Residual Connections
        self.res_block1 = ResidualBlock(12, 32, kernel_size, dropout)
        self.res_block2 = ResidualBlock(32, 32, kernel_size, dropout)
        self.pool2 = nn.AvgPool1d(kernel_size=kernel_size, stride=2)

        self.res_block3 = ResidualBlock(32, 64, kernel_size, dropout)
        self.pool3 = nn.AvgPool1d(kernel_size=kernel_size, stride=2)

        self.res_block4 = ResidualBlock(64, 64, kernel_size, dropout)
        self.res_block5 = ResidualBlock(64, 64, kernel_size, dropout)

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=64,  # CNN output channels
            hidden_size=lstm_hidden_size // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout_lstm = nn.Dropout(dropout)

        # Fully Connected Layer
        self.fc1 = nn.Linear(lstm_hidden_size, embedded_size)  
        self.dropout_fc1 = nn.Dropout(dropout * 2)

        # Contrastive Learning Projector
        self.contrastive_projector = nn.Sequential(
            nn.Linear(embedded_size, embedded_size),
            nn.ReLU(),
            nn.Linear(embedded_size, CL_embedded_size)
        )

    def forward(self, x):
        # CNN Feature Extraction with Residual Blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.pool2(x)

        x = self.res_block3(x)
        x = self.pool3(x)

        x = self.res_block4(x)
        x = self.res_block5(x)

        # Prepare for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  

        # LSTM Feature Extraction with Residual Skip Connection
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :] + x[:, -1, :]  # Residual connection

        x = self.dropout_lstm(x)

        # Fully Connected Layer
        x = self.dropout_fc1(F.leaky_relu(self.fc1(x), negative_slope=0.1))

        # Contrastive Learning Embedding
        transfered_embd = self.contrastive_projector(x)

        return x, transfered_embd  # Encoder output & contrastive embedding



########################################


class ECG_Encoder(nn.Module):
    def __init__(self, signal_length=5000, embedded_size=256, CL_embedded_size=64, kernel_size=15, dropout=0.1, alpha=0.1, num_layers=2, nhead=4, dim_feedforward=256, seed=42):
        super(ECG_Encoder, self).__init__()

        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        self.signal_length = signal_length
        padding_size = int((kernel_size - 1) / 2)
            
        self.conv1 = nn.Conv1d(12, 32, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn1 = nn.BatchNorm1d(32)
        self.avgpool1 = nn.AvgPool1d(kernel_size=kernel_size, stride=2)
        
        self.conv2 = nn.Conv1d(32, 32, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn2 = nn.BatchNorm1d(32)
        self.avgpool2 = nn.AvgPool1d(kernel_size=kernel_size, stride=2) 
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=kernel_size, stride=2, padding=padding_size)
        self.bn5 = nn.BatchNorm1d(64)

        self.flatten_size = self._get_flatten_size()
        self.fc1 = nn.Linear(self.flatten_size, 256)  

        self.alpha = alpha
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout_fc1 = nn.Dropout(dropout*2)

        # Contrastive learning projector
        self.contrastive_projector = nn.Sequential(
            nn.Linear(embedded_size, embedded_size),
            nn.ReLU(),
            nn.Linear(embedded_size, CL_embedded_size)
        )


        encoder_layers = TransformerEncoderLayer(d_model = embedded_size, nhead=nhead, dim_feedforward= embedded_size, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def _get_flatten_size(self):
        """Get the size of the flattened feature map after convolutions."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 12, self.signal_length)
            x = F.leaky_relu(self.conv1(dummy_input))
            x = F.leaky_relu(self.conv2(x))
            x = self.avgpool1(x)  
            x = F.leaky_relu(self.conv3(x))
            x = self.avgpool2(x)  
            x = F.leaky_relu(self.conv4(x))
            x = F.leaky_relu(self.conv5(x))
            return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=self.alpha))
        x = self.dropout2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=self.alpha))
        x = self.avgpool1(x)
        x = self.dropout3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=self.alpha))
        x = self.avgpool2(x)
        x = self.dropout4(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=self.alpha))
        x = self.dropout5(F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=self.alpha))
        
        x = x.permute(0, 2, 1)  # Change shape to (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling on sequence dim


        #x = self.dropout_fc1(F.leaky_relu(self.fc1(x), negative_slope=self.alpha))

        transfered_embd = self.contrastive_projector(x)  # Contrastive learning embedding
        
        return x, transfered_embd  # Encoder output & contrastive embedding








