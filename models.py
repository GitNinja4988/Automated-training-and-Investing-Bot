import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from config import *

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class PricePredictor:
    def __init__(self, input_shape):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_shape = input_shape
        
        # Initialize models
        self.lstm_model = LSTMModel(input_shape[1], LSTM_UNITS).to(self.device)
        self.gru_model = GRUModel(input_shape[1], GRU_UNITS).to(self.device)
        
        # Initialize optimizers
        self.lstm_optimizer = optim.Adam(self.lstm_model.parameters(), lr=LEARNING_RATE)
        self.gru_optimizer = optim.Adam(self.gru_model.parameters(), lr=LEARNING_RATE)
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Initialize scalers
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        # Create models directory if it doesn't exist
        self.models_dir = 'trained_models'
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def prepare_data(self, df):
        """Prepare data for training/prediction"""
        # Select features for prediction
        features = ['open', 'high', 'low', 'close', 'volume']
        data = df[features].values
        
        # Scale features
        scaled_data = self.feature_scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - SEQUENCE_LENGTH):
            X.append(scaled_data[i:(i + SEQUENCE_LENGTH)])
            y.append(scaled_data[i + SEQUENCE_LENGTH, 3])  # Predict close price
        
        return np.array(X), np.array(y)
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train both LSTM and GRU models"""
        # Convert data to PyTorch tensors
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        val_dataset = TimeSeriesDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Training history
        lstm_history = {'train_loss': [], 'val_loss': []}
        gru_history = {'train_loss': [], 'val_loss': []}
        
        best_lstm_val_loss = float('inf')
        best_gru_val_loss = float('inf')
        lstm_patience = 0
        gru_patience = 0
        
        for epoch in range(EPOCHS):
            # Train LSTM
            self.lstm_model.train()
            lstm_train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.lstm_optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.lstm_optimizer.step()
                lstm_train_loss += loss.item()
            
            # Train GRU
            self.gru_model.train()
            gru_train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.gru_optimizer.zero_grad()
                outputs = self.gru_model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.gru_optimizer.step()
                gru_train_loss += loss.item()
            
            # Validate models
            lstm_val_loss = self._validate_model(self.lstm_model, val_loader)
            gru_val_loss = self._validate_model(self.gru_model, val_loader)
            
            # Update histories
            lstm_history['train_loss'].append(lstm_train_loss / len(train_loader))
            lstm_history['val_loss'].append(lstm_val_loss)
            gru_history['train_loss'].append(gru_train_loss / len(train_loader))
            gru_history['val_loss'].append(gru_val_loss)
            
            # Early stopping
            if lstm_val_loss < best_lstm_val_loss:
                best_lstm_val_loss = lstm_val_loss
                self._save_model(self.lstm_model, 'lstm_model.pth')
                lstm_patience = 0
            else:
                lstm_patience += 1
            
            if gru_val_loss < best_gru_val_loss:
                best_gru_val_loss = gru_val_loss
                self._save_model(self.gru_model, 'gru_model.pth')
                gru_patience = 0
            else:
                gru_patience += 1
            
            # Check early stopping
            if lstm_patience >= EARLY_STOPPING_PATIENCE and gru_patience >= EARLY_STOPPING_PATIENCE:
                break
        
        return lstm_history, gru_history
    
    def _validate_model(self, model, val_loader):
        """Validate a model on validation data"""
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                total_loss += loss.item()
        return total_loss / len(val_loader)
    
    def _save_model(self, model, filename):
        """Save model to file"""
        path = os.path.join(self.models_dir, filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.lstm_optimizer.state_dict() if isinstance(model, LSTMModel) else self.gru_optimizer.state_dict(),
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }, path)
    
    def _load_model(self, model, filename):
        """Load model from file"""
        path = os.path.join(self.models_dir, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            if isinstance(model, LSTMModel):
                self.lstm_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                self.gru_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.feature_scaler = checkpoint['feature_scaler']
            self.target_scaler = checkpoint['target_scaler']
            return True
        return False
    
    def predict(self, X):
        """Make predictions using both models"""
        self.lstm_model.eval()
        self.gru_model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(X).to(self.device)
            lstm_pred = self.lstm_model(X)
            gru_pred = self.gru_model(X)
            
            # Average predictions
            combined_pred = (lstm_pred + gru_pred) / 2
            
            # Inverse transform predictions
            pred_array = combined_pred.cpu().numpy().reshape(-1, 1)
            pred_array = np.hstack([np.zeros((len(pred_array), 4)), pred_array])
            predictions = self.target_scaler.inverse_transform(pred_array)[:, 4]
            
            return predictions
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate models on test data"""
        test_dataset = TimeSeriesDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        lstm_loss = self._validate_model(self.lstm_model, test_loader)
        gru_loss = self._validate_model(self.gru_model, test_loader)
        
        return {
            'lstm': (lstm_loss,),
            'gru': (gru_loss,)
        } 