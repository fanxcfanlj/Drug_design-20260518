
# -*- coding: utf-8 -*-
"""
Drug Activity Prediction & Virtual Screening Pipeline
Function:
1. Build MLP model based on drug molecular fingerprint features
2. Train with weighted focal loss for imbalanced binary classification
3. Evaluate model with comprehensive medical classification metrics
4. Perform virtual screening on Lipinski-compliant molecular library
5. Save screened active molecular SMILES and fingerprint features
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    matthews_corrcoef
)

# ====================== Global Configurations ======================
# Random seed for reproducibility
SEED = 88

# File relative paths
TRAIN_FEATURES_PATH = "mtt_grover_vectors.npz"
TRAIN_LABELS_PATH = "mtt_data_g4ldb_valid.csv"
SCREEN_FEATURES_PATH = "Lipinski_grover_vectors.npz"
SCREEN_SMILES_PATH = "Lipinski_filtered_smiles.csv"
MODEL_SAVE_PATH = "mtt_best_model.pth"
SCREEN_SMILES_SAVE_PATH = "mtt_selected_smiles.csv"
SCREEN_FEATURES_SAVE_PATH = "mtt_selected_features.npz"

# Training hyperparameters
BATCH_SIZE = 128
EPOCHS = 500
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.8
HIDDEN_DIM = 128
# ===================================================================

# Set random seed for full reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class DrugOnlyDataset(Dataset):
    """
    Dataset class for drug molecular fingerprint features and binary labels
    """
    def __init__(self, drug_data, labels):
        self.drug_data = drug_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'drug': torch.FloatTensor(self.drug_data[idx]),
            'label': torch.FloatTensor([self.labels[idx]])
        }


class NewDrugDataset(Dataset):
    """
    Dataset class for unlabeled molecular virtual screening
    """
    def __init__(self, drug_data):
        self.drug_data = drug_data

    def __len__(self):
        return len(self.drug_data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.drug_data[idx])


class DrugOnlyModel(nn.Module):
    """
    MLP Model for drug activity binary classification
    """
    def __init__(self, input_dim, hidden=HIDDEN_DIM, dropout=DROPOUT_RATE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.Mish(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.Mish(),
            nn.Dropout(dropout * 0.8),

            nn.Linear(hidden // 2, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.Mish(),

            nn.Linear(hidden // 4, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class WeightedFocalLoss(nn.Module):
 """
    Weighted Focal Loss for imbalanced binary classification
    """
 def __init__(self, alpha=0.6, gamma=2.0, pos_weight=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # Weight positive and negative samples differently
        weight_mask = torch.where(
            targets > 0.5,
            self.pos_weight * torch.ones_like(targets),
 torch.ones_like(targets)
        )
        return (focal_loss * weight_mask).mean()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Single epoch training function"""
    model.train()
    total_loss = 0.0
    for batch in loader:
        drug = batch['drug'].to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()
        output = model(drug)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def val_one_epoch(model, loader, criterion, device):
    """Single epoch validation function"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            drug = batch['drug'].to(device)
            label = batch['label'].to(device)
            output = model(drug)
            loss = criterion(output, label)
            total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_model(model, loader, device, dataset_name="Test"):
    """Comprehensive model evaluation with multiple classification metrics"""
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in loader:
            drug = batch['drug'].to(device)
            label = batch['label'].to(device)
            output = model(drug)

            y_true.extend(label.cpu().numpy())
            y_pred.extend((output > 0.5).cpu().numpy().astype(int))
            y_prob.extend(output.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Calculate core metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Print evaluation results
    print(f"\n{dataset_name} Set Metrics:")
    print(f"Accuracy (Accuracy): {acc:.4f}")
    print(f"Precision (Precision): {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity (Specificity): {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC (Matthews Correlation Coefficient): {mcc:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load training data
    print("\nLoading training dataset...")
    drug_feature_data = np.load(TRAIN_FEATURES_PATH)['fps']
    label_df = pd.read_csv(TRAIN_LABELS_PATH)
    labels = label_df['mtt_label'].values
    print(f"Total training samples: {len(labels)}")
    print(f"Feature dimension: {drug_feature_data.shape[1]}")

    # Train/Val/Test split (stratified sampling)
    X_train, X_test, y_train, y_test = train_test_split(
        drug_feature_data, labels, test_size=0.2, stratify=labels, random_state=SEED
 )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )

    # Build datasets and dataloaders
    train_dataset = DrugOnlyDataset(X_train, y_train)
    val_dataset = DrugOnlyDataset(X_val, y_val)
    test_dataset = DrugOnlyDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss function and optimizer
    model = DrugOnlyModel(input_dim=drug_feature_data.shape[1]).to(device)
    criterion = WeightedFocalLoss(alpha=0.6, gamma=2, pos_weight=2).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Model training loop
    print("\nStart model training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = val_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Comprehensive model evaluation
    print("\n" + "="*60)
    print("Final Model Evaluation Results")
    print("="*60)
    evaluate_model(model, train_loader, device, "Training")
    evaluate_model(model, val_loader, device, "Validation")
    evaluate_model(model, test_loader, device, "Test")
    print("="*60)

    # Save trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved successfully to: {MODEL_SAVE_PATH}")

    # Virtual screening process
    print("\nStart molecular virtual screening...")
    # Load screening data
    screen_features = np.load(SCREEN_FEATURES_PATH)['fps']
    screen_smiles_df = pd.read_csv(SCREEN_SMILES_PATH)

    # Build screening dataloader
    screen_dataset = NewDrugDataset(screen_features)
    screen_loader = DataLoader(screen_dataset, batch_size=256, shuffle=False)

    # Load trained model for inference
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Predict and screen active molecules
    selected_indices = []
    with torch.no_grad():
        for batch_idx, drug in enumerate(screen_loader):
            drug = drug.to(device)
            outputs = model(drug)
            preds = (outputs > 0.5).cpu().numpy().astype(int)
            batch_selected = np.where(preds == 1)[0] + batch_idx * 256
            selected_indices.extend(batch_selected)

    # Prevent index out of bounds
    selected_indices = [idx for idx in selected_indices if idx < len(screen_smiles_df)]

    # Extract screened molecular information
    selected_smiles = screen_smiles_df.iloc[selected_indices]['SMILES']
    selected_features = screen_features[selected_indices]

    # Save screening results
    selected_smiles.to_csv(SCREEN_SMILES_SAVE_PATH, index=False)
    np.savez(SCREEN_FEATURES_SAVE_PATH, fps=selected_features)
    print(f"Virtual screening completed!")
    print(f"Screened active molecules count: {len(selected_smiles)}")
    print(f"Screened SMILES saved to: {SCREEN_SMILES_SAVE_PATH}")
    print(f"Screened features saved to: {SCREEN_FEATURES_SAVE_PATH}")


if __name__ == "__main__":
    main()