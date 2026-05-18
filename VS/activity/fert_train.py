
# -*- coding: utf-8 -*-
"""
Dual-Modality Cross-Attention Fusion Model for Fertility Activity Prediction
Model Architecture:
1. Dual-branch feature extraction (Drug molecular fingerprint + DNA embedding)
2. Multi-head self-attention for single modality feature enhancement
3. Cross-attention for inter-modal feature interaction
4. SE channel attention for adaptive feature weighting
5. Weighted Focal Loss for imbalanced binary classification
6. Two-stage molecular virtual screening pipeline
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
    precision_score, recall_score, matthews_corrcoef
)

# ====================== Global Configurations ======================
# Reproducibility seed
SEED = 42

# Relative file paths
TRAIN_DRUG_FEAT_PATH = "fert_grover_vectors.npz"
TRAIN_DNA_FEAT_PATH = "fert_dna_embeddings.npz"
TRAIN_LABEL_PATH = "fert_data_g4ldb_valid.csv"
SCREEN_DRUG_FEAT_PATH = "mtt_selected_features.npz"
SCREEN_DNA_FEAT_PATH = "dna_embeddings.npz"
SCREEN_SMILES_PATH = "mtt_selected_smiles.csv"

# Save paths
MODEL_SAVE_PATH = "fertility_best_model.pth"
SCREEN_RESULT_CSV_PATH = "fert_selected_smiles.csv"
SCREEN_RESULT_FEAT_PATH = "fert_selected_features.npz"

# Training hyperparameters
BATCH_SIZE = 128
EPOCHS = 500
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.7
ATTN_HEADS = 8
HIDDEN_DIM = 256
# ===================================================================

# Set random seed for full reproducibility
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class FertilityDataset(Dataset):
    """
    Dual-modality dataset for drug fingerprint + DNA embedding + binary label
    """
    def __init__(self, drug_data, dna_data, labels):
        self.drug_data = drug_data
        self.dna_data = dna_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'drug': torch.FloatTensor(self.drug_data[idx]),
            'dna': torch.FloatTensor(self.dna_data[idx]),
            'label': torch.FloatTensor([self.labels[idx]])
        }


class MultiHeadSelfAttention(nn.Module):
    """Single modality multi-head self-attention module"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.fc(out)
        return self.norm(out + x)


class CrossAttention(nn.Module):
    """Cross-attention module for dual-modality feature interaction"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        B, Nx, C = x.shape
        qkv = self.att.qkv(y).reshape(B, -1, 3, self.att.num_heads, self.att.head_dim).permute(2, 0, 3, 1, 4)
        k, v = qkv[1], qkv[2]
        q = self.att.qkv(x)[:, :, :C].reshape(B, Nx, self.att.num_heads, self.att.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.att.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
        out = self.att.fc(out)
        return self.norm(out + x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = channels // reduction
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
 )

    def forward(self, x):
        w = self.squeeze(x.unsqueeze(-1)).squeeze(-1)
        w = self.excitation(w).unsqueeze(-1)
        return x * w.squeeze(-1)


class AttFertilityModel(nn.Module):
    """Dual-modality cross-attention fusion fertility prediction model"""
    def __init__(self, drug_dim=3200, dna_dim=768, hidden=HIDDEN_DIM, heads=ATTN_HEADS, dropout=DROPOUT_RATE):
        super().__init__()
        # Single modality projection
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.dna_proj = nn.Sequential(
            nn.Linear(dna_dim, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)
        )

        # Self-attention branches
        self.drug_att = MultiHeadSelfAttention(hidden, heads, dropout)
        self.dna_att = MultiHeadSelfAttention(hidden, heads, dropout)

        # Cross-attention interaction
        self.cross_drug2dna = CrossAttention(hidden, heads, dropout)
        self.cross_dna2drug = CrossAttention(hidden, heads, dropout)

        # SE channel attention
        self.se_drug = SEBlock(hidden)
        self.se_dna = SEBlock(hidden)

        # Feature fusion & classification head
        self.comb = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, drug, dna):
        # Feature projection
        drug_f = self.drug_proj(drug).unsqueeze(1)
        dna_f = self.dna_proj(dna).unsqueeze(1)

        # Single modality self-attention enhancement
        drug_f = self.drug_att(drug_f)
        dna_f = self.dna_att(dna_f)

        # Cross-modal interaction
        drug_f2 = self.cross_drug2dna(drug_f, dna_f)
        dna_f2 = self.cross_dna2drug(dna_f, drug_f)

        # Residual connection
        drug_f = drug_f + drug_f2
        dna_f = dna_f + dna_f2

        # Squeeze dimension
        drug_f = drug_f.squeeze(1)
        dna_f = dna_f.squeeze(1)

        # Channel adaptive weighting
        drug_f = self.se_drug(drug_f)
        dna_f = self.se_dna(dna_f)

        # Feature fusion and prediction
        comb = torch.cat([drug_f, dna_f], dim=1)
        out = self.comb(comb)
        return torch.sigmoid(out)


class WeightedFocalLoss(nn.Module):
    """Weighted Focal Loss for imbalanced binary classification"""
    def __init__(self, alpha=0.75, gamma=3.0, pos_weight=3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        weight_mask = torch.where(
            targets > 0.5,
            self.pos_weight * torch.ones_like(targets),
            torch.ones_like(targets)
        )
        return (focal_loss * weight_mask).mean()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Single epoch training loop"""
    model.train()
    total_loss = 0.0
    for batch in loader:
        drug = batch['drug'].to(device)
        dna = batch['dna'].to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()
        output = model(drug, dna)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def val_one_epoch(model, loader, criterion, device):
    """Single epoch validation loop"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            drug = batch['drug'].to(device)
            dna = batch['dna'].to(device)
            label = batch['label'].to(device)
            output = model(drug, dna)
            loss = criterion(output, label)
            total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_detailed(model, loader, device, dataset_name="Test"):
    """Detailed classification report with per-class metrics"""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            drug = batch['drug'].to(device)
            dna = batch['dna'].to(device)
            label = batch['label'].to(device)
            output = model(drug, dna)
            y_true.extend(label.cpu().numpy())
            y_pred.extend((output > 0.5).cpu().numpy().astype(int))
            y_prob.extend(output.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'], output_dict=True)

    print(f"\n{'=' * 40}")
    print(f"{dataset_name} Set Detailed Results:")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"\nClass 0 (Negative):")
    print(f"Precision: {report['Class 0']['precision']:.4f}")
    print(f"Recall: {report['Class 0']['recall']:.4f}")
    print(f"F1-score: {report['Class 0']['f1-score']:.4f}")
    print(f"\nClass 1 (Positive):")
    print(f"Precision: {report['Class 1']['precision']:.4f}")
    print(f"Recall: {report['Class 1']['recall']:.4f}")
    print(f"F1-score: {report['Class 1']['f1-score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print('=' * 40)


def evaluate_simple(model, loader, device, dataset_name="Test"):
    """Simplified evaluation with comprehensive medical metrics"""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            drug = batch['drug'].to(device)
            dna = batch['dna'].to(device)
            label = batch['label'].to(device)
            output = model(drug, dna)
            y_true.extend(label.cpu().numpy())
            y_pred.extend((output > 0.5).cpu().numpy().astype(int))
            y_prob.extend(output.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\n{dataset_name} Set Metrics:")
    print(f"Accuracy (准确度): {acc:.4f}")
    print(f"Precision (精密度): {precision:.4f}")
    print(f"Recall (灵敏度): {recall:.4f}")
    print(f"Specificity (特异性): {specificity:.4f}")
    print(f"F1 Score (F测量): {f1:.4f}")
    print(f"MCC (马修相关系数): {mcc:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def virtual_screening(model, device):
    """Two-stage molecular virtual screening based on trained model"""
    # Load screening dataset
    screen_drug_feat = np.load(SCREEN_DRUG_FEAT_PATH)['fps']
    screen_dna_feat = np.load(SCREEN_DNA_FEAT_PATH)['embeddings']
    screen_smiles_df = pd.read_csv(SCREEN_SMILES_PATH)

    results = []
    selected_features = []

    # Screen with fixed first DNA embedding
    for drug_idx in range(len(screen_drug_feat)):
        print(f"Processing drug {drug_idx}/{len(screen_drug_feat)}")
        drug_tensor = torch.FloatTensor(screen_drug_feat[drug_idx]).unsqueeze(0).to(device)
        dna_tensor = torch.FloatTensor(screen_dna_feat[0]).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(drug_tensor, dna_tensor)
            pred = (output > 0.5).item()
            prob = output.item()

        if pred == 1:
            smiles = screen_smiles_df.iloc[drug_idx]['smiles']
            results.append({
                'drug_index': drug_idx,
                'dna_index': 0,
                'smiles': smiles,
                'probability': prob,
                'prediction': pred
            })
            selected_features.append(screen_drug_feat[drug_idx])
            print(f"Drug {drug_idx} passed first DNA check")

    # Save screening results
    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(SCREEN_RESULT_CSV_PATH, index=False)
        selected_features = np.array(selected_features)
        np.savez(SCREEN_RESULT_FEAT_PATH, fps=selected_features)
        print(f"\nScreening completed! Valid molecules count: {len(results)}")
        print(f"Screened SMILES saved to: {SCREEN_RESULT_CSV_PATH}")
        print(f"Screened features saved to: {SCREEN_RESULT_FEAT_PATH}")
    else:
        print("\nNo valid molecules passed the screening threshold")


def main():
    # Device initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load training data
    print("\nLoading training dataset...")
    drug_feature_data = np.load(TRAIN_DRUG_FEAT_PATH)['fps']
    dna_feature_data = np.load(TRAIN_DNA_FEAT_PATH)['embeddings']
    label_df = pd.read_csv(TRAIN_LABEL_PATH)
    labels = label_df['fert_label'].values
    print(f"Total training samples: {len(labels)}")
    print(f"Drug feature dimension: {drug_feature_data.shape[1]}")
    print(f"DNA embedding dimension: {dna_feature_data.shape[1]}")

    # Stratified dataset split
    X_train_drug, X_test_drug, X_train_dna, X_test_dna, y_train, y_test = train_test_split(
        drug_feature_data, dna_feature_data, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    X_train_drug, X_val_drug, X_train_dna, X_val_dna, y_train, y_val = train_test_split(
        X_train_drug, X_train_dna, y_train, test_size=0.2, stratify=y_train, random_state=SEED
    )

    # Build dataloaders
    train_dataset = FertilityDataset(X_train_drug, X_train_dna, y_train)
    val_dataset = FertilityDataset(X_val_drug, X_val_dna, y_val)
    test_dataset = FertilityDataset(X_test_drug, X_test_dna, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model initialization
    model = AttFertilityModel(drug_feature_data.shape[1], dna_feature_data.shape[1]).to(device)
    criterion = WeightedFocalLoss(alpha=0.75, gamma=3.0, pos_weight=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    print("\nStart model training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = val_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Detailed evaluation
    print("\n" + "="*60)
    print("Final Detailed Classification Results")
    print("="*60)
    evaluate_detailed(model, train_loader, device, "Training")
    evaluate_detailed(model, val_loader, device, "Validation")
    evaluate_detailed(model, test_loader, device, "Test")

    # Simplified metric evaluation
    print("\n" + "="*40)
    print("Simplified Comprehensive Metrics Results")
    print("="*40)
    evaluate_simple(model, train_loader, device, "Training")
    evaluate_simple(model, val_loader, device, "Validation")
    evaluate_simple(model, test_loader, device, "Test")
    print("="*40)

    # Save trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved successfully to: {MODEL_SAVE_PATH}")

    # Execute virtual screening
    virtual_screening(model, device)


if __name__ == "__main__":
    main()