
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
import random
from pathlib import Path
import copy

# ==================================================================================================
# 1. MODEL DEFINITIONS (Self-Contained)
# ==================================================================================================

class GEI_VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(GEI_VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64)
        
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(128)
        
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(256)
        
        self.flatten_size = 256 * 8 * 4
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, self.flatten_size)
        
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(128)
        
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(64)
        
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn3 = nn.BatchNorm2d(32)
        
        self.dec_conv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.view(x.size(0), 256, 8, 4)
        x = self.relu(self.dec_bn1(self.dec_conv1(x)))
        x = self.relu(self.dec_bn2(self.dec_conv2(x)))
        x = self.relu(self.dec_bn3(self.dec_conv3(x)))
        x = self.sigmoid(self.dec_conv4(x))
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

class ContrastiveVAE(nn.Module):
    def __init__(self, latent_dim=128, projection_dim=128):
        super(ContrastiveVAE, self).__init__()
        self.vae = GEI_VAE(latent_dim=latent_dim)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, projection_dim)
        )

    def encode(self, x):
        return self.vae.encode(x)
        
    def reparameterize(self, mu, log_var):
        return self.vae.reparameterize(mu, log_var)
        
    def decode(self, z):
        return self.vae.decode(z)

    def forward(self, x, return_projection=False):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        
        if return_projection:
            projection = self.projection_head(mu)
            return reconstruction, mu, log_var, projection
        else:
            return reconstruction, mu, log_var

def vae_loss(reconstruction, target, mu, log_var, beta=1.0):
    recon_loss = nn.functional.mse_loss(reconstruction, target, reduction='sum') / target.size(0)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / target.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def contrastive_loss_fn(z_i, z_j, temperature=0.07):
    # NT-Xent loss
    batch_size = z_i.size(0)
    z_i = nn.functional.normalize(z_i, dim=1)
    z_j = nn.functional.normalize(z_j, dim=1)
    
    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    
    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    
    # Positive pairs are (i, i+batch_size) and (i+batch_size, i)
    labels = torch.arange(batch_size, device=z_i.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    similarity_matrix = similarity_matrix / temperature
    criterion = nn.CrossEntropyLoss()
    loss = criterion(similarity_matrix, labels)
    return loss

# ==================================================================================================
# 2. DATASET DEFINITION (Full GEI Only)
# ==================================================================================================

class FullGEIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Path to 'Pathology_dataset' (nested structure)
        Uses ONLY '*full.jpg' images.
        """
        self.root_dir = list(Path(root_dir).glob("**/*full.jpg"))
        self.transform = transform
        self.samples = []
        self.labels = []
        self.conditions = []
        
        # Map conditions to integers
        # Assuming structure: root / Condition / Subject / ...
        # Known conditions: diplegic, hemiplegic, neuropathic, normal, parkinson (from dir listing)
        self.class_map = {
            'normal': 0,
            'diplegic': 1, 
            'neuropathic': 2,
            'hemiplegic': 3,
            'parkinson': 4
        }
        
        print(f"Scanning for full.jpg in {root_dir}...")
        
        for img_path in self.root_dir:
            # Extract condition from path
            # Example path: .../Pathology_dataset/Parkinson/s12/GEIs/.../full.jpg
            # We need to robustly find the condition name
            parts = img_path.parts
            condition = None
            for part in parts:
                if part.lower() in self.class_map:
                    condition = part.lower()
                    break
            
            if condition:
                self.samples.append(str(img_path))
                self.labels.append(self.class_map[condition])
                self.conditions.append(condition)
                
        print(f"Found {len(self.samples)} 'full.jpg' GEIs across {len(set(self.labels))} classes.")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('L') # Grayscale
            image = image.resize((64, 128)) # Standard GEI size (W, H)
            
            # Normalize to [0, 1]
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0) # (1, 128, 64)
            
            return image, label, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((1, 128, 64)), label, img_path

# ==================================================================================================
# 3. TRAINING & EVALUATION UTILS
# ==================================================================================================

def train_model(model, train_loader, epochs=50, lr=1e-4, contrastive=False):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            images, _, _ = data
            images = images.to(device)
            
            optimizer.zero_grad()
            
            if contrastive:
                # Contrastive requires two views. Simple augmentation: Image + Noisy Image or same image
                # Here we simulate self-contrastive by passing the same image twice (weak augmentation)
                # Ideally we would have real augmentations, but for "experiments_final" replication logic:
                recon, mu, log_var, proj1 = model(images, return_projection=True)
                _, _, _, proj2 = model(images, return_projection=True) # Usually different augment needed
                
                v_loss, _, _ = vae_loss(recon, images, mu, log_var)
                c_loss = contrastive_loss_fn(proj1, proj2)
                loss = v_loss + 0.1 * c_loss # Weighting from original script
            else:
                recon, mu, log_var = model(images)
                loss, _, _ = vae_loss(recon, images, mu, log_var)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    return model

def evaluate_model(model, data_loader, run_name):
    model.eval()
    embeddings = []
    labels = []
    
    # Reconstruction metrics
    total_recon_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, lbls, _ in data_loader:
            images = images.to(device)
            recon, mu, log_var = model(images) if not isinstance(model, ContrastiveVAE) else model(images, return_projection=False)[:3]
            
            # Reconstruction MSE
            mse = nn.functional.mse_loss(recon, images, reduction='sum').item()
            total_recon_loss += mse
            total_samples += images.size(0)
            
            # Store embeddings (mu)
            embeddings.append(mu.cpu().numpy())
            labels.append(lbls.numpy())
            
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    avg_mse = total_recon_loss / total_samples
    
    # Classification (KNN)
    # Split embeddings for simple eval (80/20 split on extracted features)
    # Note: Proper ML would split subjects, but for quick embedding quality check:
    indices = np.arange(len(embeddings))
    np.random.shuffle(indices)
    split = int(0.8 * len(embeddings))
    train_idx, test_idx = indices[:split], indices[split:]
    
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    
    print(f"--- {run_name} Results ---")
    print(f"Reconstruction MSE: {avg_mse:.4f}")
    print(f"KNN Accuracy: {acc*100:.2f}%")
    
    return avg_mse, acc

# ==================================================================================================
# 4. MAIN PIPELINE
# ==================================================================================================

if __name__ == "__main__":
    
    # CONFIG
    # Look for Pathology_dataset relative to this script
    # Script is in experiments_final/. Dataset is likely ONE LEVEL UP from experiments_final in project root
    # Adjust path if needed
    SCRIPT_DIR = Path(__file__).parent
    # Script is in experiments_final/full_gei_analysis/
    # Dataset is in Pathology_dataset (root) -> up 3 levels
    DATA_ROOT = SCRIPT_DIR.parent.parent / "Pathology_dataset"
    CHECKPOINT_DIR = SCRIPT_DIR.parent.parent / "experiments_final" / "checkpoints"
    
    device = torch.device('cpu') # Use CPU for mac unless mps is verified
    BATCH_SIZE = 16
    EPOCHS = 30 # Reduced for speed in this demo, original was ~100
    
    print(f"Running Full-GEI Pipeline.")
    print(f"Data Root: {DATA_ROOT}")
    
    # 1. LOAD DATA
    dataset = FullGEIDataset(DATA_ROOT)
    if len(dataset) == 0:
        print("CRITICAL ERROR: No 'full.jpg' images found. Check path.")
        exit(1)
        
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    results = {}
    
    # ============================================================
    # RUN 1: EXP3 (FROM SCRATCH)
    # ============================================================
    print("\n\n=== RUN 1: EXP3 (Train from Scratch on Full GEIs) ===")
    
    # VAE Scratch
    vae_scratch = GEI_VAE().to(device)
    print("Training VAE (Scratch)...")
    vae_scratch = train_model(vae_scratch, loader, epochs=EPOCHS, contrastive=False)
    mse, acc = evaluate_model(vae_scratch, loader, "EXP3 VAE")
    results['EXP3_VAE'] = {'MSE': mse, 'ACC': acc}
    
    # Contrastive Scratch
    cvae_scratch = ContrastiveVAE().to(device)
    print("Training Contrastive VAE (Scratch)...")
    cvae_scratch = train_model(cvae_scratch, loader, epochs=EPOCHS, contrastive=True)
    mse, acc = evaluate_model(cvae_scratch, loader, "EXP3 Contrastive")
    results['EXP3_Contrastive'] = {'MSE': mse, 'ACC': acc}


    # ============================================================
    # RUN 2: EXP2 (FINETUNE)
    # ============================================================
    print("\n\n=== RUN 2: EXP2 (Finetune from CASIA on Full GEIs) ===")
    
    if (CHECKPOINT_DIR / "exp1_vae_casia.pth").exists():
        # VAE Finetune
        vae_ft = GEI_VAE().to(device)
        ckpt = torch.load(CHECKPOINT_DIR / "exp1_vae_casia.pth", map_location=device)
        # Handle dictionary wrap if present
        if 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        vae_ft.load_state_dict(ckpt)
        print("Loaded CASIA VAE weights. Finetuning...")
        vae_ft = train_model(vae_ft, loader, epochs=EPOCHS, lr=1e-5, contrastive=False) # Lower LR for finetune
        mse, acc = evaluate_model(vae_ft, loader, "EXP2 VAE (Finetune)")
        results['EXP2_VAE'] = {'MSE': mse, 'ACC': acc}
    else:
        print("Skipping EXP2 VAE (No checkpoint found)")
        
    if (CHECKPOINT_DIR / "exp1_contrastive_casia.pth").exists():
        # Contrastive Finetune
        cvae_ft = ContrastiveVAE().to(device)
        ckpt = torch.load(CHECKPOINT_DIR / "exp1_contrastive_casia.pth", map_location=device)
        if 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        # Note: CASIA Contrastive weights might have key mismatch if projection head changed. Strict=False helps.
        cvae_ft.load_state_dict(ckpt, strict=False)
        print("Loaded CASIA Contrastive weights. Finetuning...")
        cvae_ft = train_model(cvae_ft, loader, epochs=EPOCHS, lr=1e-5, contrastive=True)
        mse, acc = evaluate_model(cvae_ft, loader, "EXP2 Contrastive (Finetune)")
        results['EXP2_Contrastive'] = {'MSE': mse, 'ACC': acc}
    else:
        print("Skipping EXP2 Contrastive (No checkpoint found)")


    # ============================================================
    # RUN 3: EXP1 (ZERO-SHOT EVALUATION)
    # ============================================================
    print("\n\n=== RUN 3: EXP1 (Zero-Shot CASIA on Full GEIs) ===")
    
    if (CHECKPOINT_DIR / "exp1_vae_casia.pth").exists():
        vae_zs = GEI_VAE().to(device)
        ckpt = torch.load(CHECKPOINT_DIR / "exp1_vae_casia.pth", map_location=device)
        if 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        vae_zs.load_state_dict(ckpt)
        print("Evaluating CASIA VAE (Zero-Shot)...")
        mse, acc = evaluate_model(vae_zs, loader, "EXP1 VAE (Zero-Shot)")
        results['EXP1_VAE'] = {'MSE': mse, 'ACC': acc}
    
    if (CHECKPOINT_DIR / "exp1_contrastive_casia.pth").exists():
        cvae_zs = ContrastiveVAE().to(device)
        ckpt = torch.load(CHECKPOINT_DIR / "exp1_contrastive_casia.pth", map_location=device)
        if 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        cvae_zs.load_state_dict(ckpt, strict=False)
        print("Evaluating CASIA Contrastive (Zero-Shot)...")
        mse, acc = evaluate_model(cvae_zs, loader, "EXP1 Contrastive (Zero-Shot)")
        results['EXP1_Contrastive'] = {'MSE': mse, 'ACC': acc}


    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n\n" + "="*50)
    print("FINAL RESULTS TABLE (Data: ONLY *full.jpg)")
    print("="*50)
    print(f"{'Experiment':<30} | {'MSE':<10} | {'Accuracy':<10}")
    print("-" * 56)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['MSE']:.4f}     | {metrics['ACC']*100:.2f}%")
