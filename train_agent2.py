# train_agent2.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Moksha_1.models.deep_quant.dataset import MokshaDataset
from Moksha_1.models.deep_quant.model import FinancialNN3

def train_model():
    # 1. Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 10  # Start small
    
    # 2. Prepare Data
    print("⏳ Initializing Agent 2 Data Pipeline...")
    dataset = MokshaDataset() # Loads ALL symbols from DB
    
    if len(dataset) == 0:
        print("❌ Dataset empty. Check feature generation.")
        return

    # Train/Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 3. Initialize Model
    input_dim = dataset.X.shape[1]
    print(f"🏗️ Building NN3 Model (Input Dim: {input_dim})...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on: {device}")
    
    model = FinancialNN3(input_dim=input_dim).to(device)
    
    # Loss Function: MSE (Mean Squared Error) is standard for regression
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # L2 Regularization
    
    # 4. Training Loop
    print("\n--- 🏁 STARTING TRAINING ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                preds = model(X_val)
                val_loss += criterion(preds, y_val).item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 5. Save the Brain
    torch.save(model.state_dict(), "moksha_nn3.pth")
    print("\n✅ Model Saved: moksha_nn3.pth")

if __name__ == "__main__":
    train_model()