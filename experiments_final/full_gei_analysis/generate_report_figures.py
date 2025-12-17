import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sys
from pathlib import Path

# Import pipeline definitions
sys.path.append(str(Path(__file__).parent.resolve()))
from run_full_gei_pipeline import FullGEIDataset, create_subject_aware_split, GEI_VAE, ContrastiveVAE, extract_features

def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('EXP2 Contrastive VAE Confusion Matrix') # EXP2 had best KNN Acc
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def plot_f1_scores(y_true, y_pred, classes, output_path):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    f1s = [report[c]['f1-score'] for c in classes]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, f1s, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f'])
    plt.ylim(0, 1.05)
    plt.ylabel('F1 Score')
    plt.title('Per-Class F1 Scores (EXP2 Contrastive VAE)')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def plot_roc_curve(y_binary_true, y_prob, output_path):
    fpr, tpr, _ = roc_curve(y_binary_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'EXP3 VAE (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Binary Anomaly Detection ROC')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def main():
    # CONFIG
    SEED = 42
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    SCRIPT_DIR = Path(__file__).parent
    DATA_ROOT = SCRIPT_DIR.parent.parent / "Pathology_dataset"
    OUTPUT_DIR = SCRIPT_DIR / "figures"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    device = torch.device('cpu')
    
    # FIX: Inject device into imported module scope because extract_features relies on global 'device'
    import run_full_gei_pipeline
    run_full_gei_pipeline.device = device
    
    # Load Data with Strict Split
    dataset = FullGEIDataset(DATA_ROOT)
    train_set, test_set = create_subject_aware_split(dataset, test_ratio=0.2, seed=SEED)
    
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, generator=g)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, generator=g)
    
    # Only keep classes that actually exist in the dataset
    present_labels = sorted(list(set(dataset.labels)))
    # Reverse lookup for names
    classes = []
    inv_map = {v: k for k, v in dataset.class_map.items()}
    for lbl in present_labels:
        classes.append(inv_map[lbl])
    
    # --- 1. Multi-Class Analysis (Using EXP2 Contrastive - Best KNN) ---
    # Since we can't load the just-trained model from memory (it wasn't saved to disk in the run script except internally)
    # We will TRAIN it quickly to reproduce the exact state (Seed 42 guarantees identical weights)
    
    print("Retraining EXP2 Contrastive to reproduce features for plotting...")
    model = ContrastiveVAE().to(device)
    # Load CASIA Pretrain
    ckpt = torch.load(SCRIPT_DIR.parent / "checkpoints" / "exp1_contrastive_casia.pth", map_location=device)
    if 'model_state_dict' in ckpt: ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt, strict=False)
    
    # Train Loop (Same as run_full_gei_pipeline)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    augmenter = T.RandomAffine(degrees=10, translate=(0.05, 0.05))
    
    # Retraining for 30 epochs
    from run_full_gei_pipeline import vae_loss, contrastive_loss_fn
    for epoch in range(30):
        for data in train_loader:
             images, _, _ = data
             images = images.to(device)
             optimizer.zero_grad()
             # Contrastive Pass
             recon, mu, log_var, proj1 = model(images, return_projection=True)
             images_aug = augmenter(images)
             _, _, _, proj2 = model(images_aug, return_projection=True)
             
             v_loss, _, _ = vae_loss(recon, images, mu, log_var)
             c_loss = contrastive_loss_fn(proj1, proj2)
             loss = v_loss + 0.1 * c_loss
             loss.backward()
             optimizer.step()
    
    # Extract Features
    X_train, y_train, _, _ = extract_features(model, train_loader)
    X_test, y_test, _, _ = extract_features(model, test_loader)
    
    # Train KNN (Best Classifier)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Plot 1 & 2
    plot_confusion_matrix(y_test, y_pred, classes, OUTPUT_DIR / "fig7_confusion_matrix.png")
    plot_f1_scores(y_test, y_pred, classes, OUTPUT_DIR / "fig8_f1_scores.png")

    # --- 2. Binary Analysis (Using EXP3 VAE - Best Binary) ---
    print("Training EXP3 VAE for Binary ROC...")
    model_bin = GEI_VAE().to(device)
    optimizer = torch.optim.Adam(model_bin.parameters(), lr=1e-4)
    model_bin.train()
    
    for epoch in range(30):
        for data in train_loader:
             images, _, _ = data
             images = images.to(device)
             optimizer.zero_grad()
             recon, mu, log_var = model_bin(images)
             loss, _, _ = vae_loss(recon, images, mu, log_var)
             loss.backward()
             optimizer.step()
             
    X_train_b, y_train_b, _, _ = extract_features(model_bin, train_loader)
    X_test_b, y_test_b, _, _ = extract_features(model_bin, test_loader)
    
    # Binary Labels
    y_train_bin = (y_train_b != 0).astype(int)
    y_test_bin = (y_test_b != 0).astype(int)
    
    # Logistic Regression for Probabilities
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_b, y_train_bin)
    y_probs = clf.predict_proba(X_test_b)[:, 1]
    
    # Plot 3
    plot_roc_curve(y_test_bin, y_probs, OUTPUT_DIR / "fig9_roc_curve.png")

if __name__ == "__main__":
    main()
