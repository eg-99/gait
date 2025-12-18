
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import random
from pathlib import Path
import copy
import torchvision.transforms as T

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
        self.subjects = [] # Store subject IDs
        
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
            # Extract condition and subject from path
            # Example path: .../Pathology_dataset/Parkinson/s12/GEIs/.../full.jpg
            parts = img_path.parts
            condition = None
            subject = None
            
            # Find condition and subject (assume subject is child of condition)
            for i, part in enumerate(parts):
                if part.lower() in self.class_map:
                    condition = part.lower()
                    if i + 1 < len(parts):
                        subject = parts[i+1] # s12, s18, etc.
                    break
            
            if condition:
                self.samples.append(str(img_path))
                self.labels.append(self.class_map[condition])
                self.conditions.append(condition)
                self.subjects.append(subject if subject else "unknown")
                
        print(f"Found {len(self.samples)} 'full.jpg' GEIs across {len(set(self.labels))} classes.")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        subject = self.subjects[idx]
        
        try:
            image = Image.open(img_path).convert('L') # Grayscale
            image = image.resize((64, 128)) # Standard GEI size (W, H)
            
            # Normalize to [0, 1]
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0) # (1, 128, 64)
            
            return image, label, subject
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((1, 128, 64)), label, subject

# ==================================================================================================
# 3. TRAINING & EVALUATION UTILS
# ==================================================================================================

def train_model(model, train_loader, epochs=50, lr=1e-4, contrastive=False):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Augmentation for Contrastive Learning (View 2)
    # We apply slight affine consistency (rotate, translate) to simulate different gait cycles
    augmenter = T.RandomAffine(degrees=10, translate=(0.05, 0.05))
    
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            images, _, _ = data
            images = images.to(device)
            
            optimizer.zero_grad()
            
            if contrastive:
                # View 1: Original Image
                recon, mu, log_var, proj1 = model(images, return_projection=True)
                
                # View 2: Augmented Image (simulate different viewing condition/cycle)
                images_aug = augmenter(images)
                _, _, _, proj2 = model(images_aug, return_projection=True)
                
                v_loss, _, _ = vae_loss(recon, images, mu, log_var)
                c_loss = contrastive_loss_fn(proj1, proj2)
                loss = v_loss + 0.1 * c_loss
            else:
                recon, mu, log_var = model(images)
                loss, _, _ = vae_loss(recon, images, mu, log_var)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    return model

def extract_features(model, loader):
    model.eval()
    embeddings = []
    labels = []
    subjects = []
    recon_loss = 0
    total = 0
    
    with torch.no_grad():
        for images, lbls, subjs in loader:
            images = images.to(device)
            recon, mu, log_var = model(images) if not isinstance(model, ContrastiveVAE) else model(images, return_projection=False)[:3]
            
            mse = nn.functional.mse_loss(recon, images, reduction='sum').item()
            recon_loss += mse
            total += images.size(0)
            
            embeddings.append(mu.cpu().numpy())
            labels.append(lbls.numpy())
            subjects.extend(list(subjs))
            
    return np.concatenate(embeddings), np.concatenate(labels), np.array(subjects), recon_loss / total

def evaluate_model(model, train_loader, test_loader, run_name):
    # Extract Train Features (for fitting classifiers)
    X_train, y_train, s_train, _ = extract_features(model, train_loader)
    
    # Extract Test Features (for evaluation)
    X_test, y_test, s_test, test_mse = extract_features(model, test_loader)
    
    # 1. Multi-Class KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred_knn)
    knn_f1 = f1_score(y_test, y_pred_knn, average='macro')
    
    # 2. Multi-Class SVM
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm, average='macro')
    
    # 3. Binary Anomaly Detection
    y_bin_train = (y_train != 0).astype(int)
    y_bin_test = (y_test != 0).astype(int)
    
    bin_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    bin_clf.fit(X_train, y_bin_train)
    y_bin_pred = bin_clf.predict(X_test)
    bin_acc = accuracy_score(y_bin_test, y_bin_pred)
    bin_f1 = f1_score(y_bin_test, y_bin_pred, average='binary')
    
    # 4. Subject Identification
    # Filter for subjects present in both train and test (otherwise can't predict)
    common_subjects = set(s_train).intersection(set(s_test))
    if len(common_subjects) > 1:
        # Create masks
        train_mask = np.isin(s_train, list(common_subjects))
        test_mask = np.isin(s_test, list(common_subjects))
        
        X_subj_train = X_train[train_mask]
        y_subj_train = LabelEncoder().fit_transform(s_train[train_mask])
        
        X_subj_test = X_test[test_mask]
        # Re-encode test to ensure matching labels (careful here)
        le = LabelEncoder()
        le.fit(s_train[train_mask]) # Fit on train subjects
        # Only keep test samples that are known subjects
        known_test_mask = np.isin(s_test[test_mask], le.classes_)
        if known_test_mask.sum() > 0:
            X_subj_test_final = X_subj_test[known_test_mask]
            y_subj_test_final = le.transform(s_test[test_mask][known_test_mask])
            
            clf_subj = LogisticRegression(max_iter=2000, multi_class='multinomial')
            clf_subj.fit(X_subj_train, y_subj_train)
            subj_acc = accuracy_score(y_subj_test_final, clf_subj.predict(X_subj_test_final))
        else:
            subj_acc = 0.0
    else:
        subj_acc = 0.0

    print(f"--- {run_name} Results ---")
    print(f"MSE: {test_mse:.2f} | KNN: {knn_acc:.2f} | SVM: {svm_acc:.2f} | Binary: {bin_acc:.2f} | Subj: {subj_acc:.2f}")
    
    return {
        'MSE': test_mse,
        'KNN_Acc': knn_acc, 'KNN_F1': knn_f1,
        'SVM_Acc': svm_acc, 'SVM_F1': svm_f1,
        'Binary_Acc': bin_acc, 'Binary_F1': bin_f1,
        'Subj_Acc': subj_acc
    }

# ==================================================================================================
# 4. MAIN PIPELINE
# ==================================================================================================

if __name__ == "__main__":
    
    # CONFIG
    SCRIPT_DIR = Path(__file__).parent
    DATA_ROOT = SCRIPT_DIR.parent.parent / "pathology_data_organized"
    CHECKPOINT_DIR = SCRIPT_DIR.parent.parent / "experiments_final" / "checkpoints"
    
    device = torch.device('cpu') 
    BATCH_SIZE = 16
    EPOCHS = 30
    
    print(f"Running Full-GEI Pipeline (Rigorous Split 80/20).")
    print(f"Data Root: {DATA_ROOT}")
    
    # 1. LOAD DATA
    dataset = FullGEIDataset(DATA_ROOT)
    if len(dataset) == 0:
        print("CRITICAL ERROR: No 'full.jpg' images found.")
        exit(1)
    
# ... (Previous code remains)

def create_subject_aware_split(dataset, test_ratio=0.2, seed=42):
    """
    Splits the dataset into train and test sets ensuring no subject overlap.
    Stratified by Class (Condition).
    """
    # Group samples by (Condition, Subject)
    # Structure: condition_map = { 'normal': ['s1', 's2'], 'diplegic': ['s5', 's6'] }
    condition_to_subjects = {}
    
    # We need to iterate the dataset to map subjects
    # Accessing internal lists directly for speed (dataset.samples, dataset.subjects, dataset.labels)
    # dataset.subjects is a list aligned with dataset.samples
    
    unique_class_labels = set(dataset.labels)
    
    # Map label -> set of subjects
    label_to_subjects = {}
    for idx, label in enumerate(dataset.labels):
        subj = dataset.subjects[idx]
        if label not in label_to_subjects:
            label_to_subjects[label] = set()
        label_to_subjects[label].add(subj)
        
    train_indices = []
    test_indices = []
    
    rng = random.Random(seed)
    
    print(f"\n--- Subject-Aware Splitting (Test Ratio={test_ratio}) ---")
    
    for label, subjects in label_to_subjects.items():
        subjects = list(subjects)
        subjects.sort() # Ensure deterministic order before shuffle
        rng.shuffle(subjects)
        
        n_test = max(1, int(len(subjects) * test_ratio)) # Ensure at least 1 test subject per class if possible
        n_train = len(subjects) - n_test
        
        train_subjs = set(subjects[n_test:])
        test_subjs = set(subjects[:n_test])
        
        # Verify strict separation
        assert train_subjs.isdisjoint(test_subjs)
        
        # Find indices for these subjects
        # This is O(N*C), acceptable for N=400
        for idx, (s, l) in enumerate(zip(dataset.subjects, dataset.labels)):
            if l == label:
                if s in train_subjs:
                    train_indices.append(idx)
                elif s in test_subjs:
                    test_indices.append(idx)
        
        condition_name = [k for k, v in dataset.class_map.items() if v == label][0]
        print(f"Class '{condition_name}': {len(train_subjs)} Train Subjs, {len(test_subjs)} Test Subjs")

    print(f"Total: {len(train_indices)} Train Images, {len(test_indices)} Test Images")
    
    # Create Subset objects
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_subset, test_subset

if __name__ == "__main__":
    
    # CONFIG
    # GLOBAL SEEDING for Perfection/Consistency
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    SCRIPT_DIR = Path(__file__).parent
    DATA_ROOT = SCRIPT_DIR.parent.parent / "pathology_data_organized"
    CHECKPOINT_DIR = SCRIPT_DIR.parent.parent / "experiments_final" / "checkpoints"
    
    device = torch.device('cpu') 
    BATCH_SIZE = 16
    EPOCHS = 30
    
    print(f"Running Full-GEI Pipeline (Perfect: Subject-Aware + Augmentation + Fixed Seed).")
    print(f"Data Root: {DATA_ROOT}")
    
    # 1. LOAD DATA
    dataset = FullGEIDataset(DATA_ROOT)
    if len(dataset) == 0:
        print("CRITICAL ERROR: No 'full.jpg' images found.")
        exit(1)
    
    # 2. Strict SUBJECT-AWARE Split
    # This prevents "Subject Leakage" (learning the person instead of the disease)
    train_set, test_set = create_subject_aware_split(dataset, test_ratio=0.2, seed=SEED)
    
    # Fixed generator for dataloaders
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, generator=g)
    
    results = {}
    
    # EXP 3
    print("\n\n=== RUN 1: EXP3 (Train from Scratch) ===")
    vae_scratch = GEI_VAE().to(device)
    vae_scratch = train_model(vae_scratch, train_loader, epochs=EPOCHS, contrastive=False)
    results['EXP3 VAE'] = evaluate_model(vae_scratch, train_loader, test_loader, "EXP3 VAE")
    
    cvae_scratch = ContrastiveVAE().to(device)
    cvae_scratch = train_model(cvae_scratch, train_loader, epochs=EPOCHS, contrastive=True)
    results['EXP3 Contrastive'] = evaluate_model(cvae_scratch, train_loader, test_loader, "EXP3 Contrastive")

    # EXP 2
    print("\n\n=== RUN 2: EXP2 (Finetune) ===")
    if (CHECKPOINT_DIR / "exp1_vae_casia.pth").exists():
        vae_ft = GEI_VAE().to(device)
        ckpt = torch.load(CHECKPOINT_DIR / "exp1_vae_casia.pth", map_location=device)
        if 'model_state_dict' in ckpt: ckpt = ckpt['model_state_dict']
        vae_ft.load_state_dict(ckpt)
        vae_ft = train_model(vae_ft, train_loader, epochs=EPOCHS, lr=1e-5, contrastive=False)
        results['EXP2 VAE'] = evaluate_model(vae_ft, train_loader, test_loader, "EXP2 VAE")
    else:
        print("Skipping EXP2 VAE (No checkpoint)")
    
    if (CHECKPOINT_DIR / "exp1_contrastive_casia.pth").exists():
        cvae_ft = ContrastiveVAE().to(device)
        ckpt = torch.load(CHECKPOINT_DIR / "exp1_contrastive_casia.pth", map_location=device)
        if 'model_state_dict' in ckpt: ckpt = ckpt['model_state_dict']
        cvae_ft.load_state_dict(ckpt, strict=False)
        cvae_ft = train_model(cvae_ft, train_loader, epochs=EPOCHS, lr=1e-5, contrastive=True)
        results['EXP2 Contrastive'] = evaluate_model(cvae_ft, train_loader, test_loader, "EXP2 Contrastive")
    else:
        print("Skipping EXP2 Contrastive (No checkpoint)")

    # EXP 1
    print("\n\n=== RUN 3: EXP1 (Zero-Shot) ===")
    if (CHECKPOINT_DIR / "exp1_vae_casia.pth").exists():
        vae_zs = GEI_VAE().to(device)
        ckpt = torch.load(CHECKPOINT_DIR / "exp1_vae_casia.pth", map_location=device)
        if 'model_state_dict' in ckpt: ckpt = ckpt['model_state_dict']
        vae_zs.load_state_dict(ckpt)
        results['EXP1 VAE'] = evaluate_model(vae_zs, train_loader, test_loader, "EXP1 VAE")
    
    if (CHECKPOINT_DIR / "exp1_contrastive_casia.pth").exists():
        cvae_zs = ContrastiveVAE().to(device)
        ckpt = torch.load(CHECKPOINT_DIR / "exp1_contrastive_casia.pth", map_location=device)
        if 'model_state_dict' in ckpt: ckpt = ckpt['model_state_dict']
        cvae_zs.load_state_dict(ckpt, strict=False)
        results['EXP1 Contrastive'] = evaluate_model(cvae_zs, train_loader, test_loader, "EXP1 Contrastive")

    print("\n\n" + "="*145)
    print("COMPREHENSIVE SUMMARY TABLE (Subject-Aware Split)")
    print("="*145)
    print(f"{'Experiment':<20} | {'MSE':<8} | {'KNN Acc':<8} {'KNN F1':<8} | {'SVM Acc':<8} {'SVM F1':<8} | {'Bin Acc':<8} {'Bin F1':<8} | {'Subj Acc':<8}")
    print("-" * 145)
    for name, m in results.items():
        print(f"{name:<20} | {m['MSE']:<8.2f} | {m['KNN_Acc']:<8.3f} {m['KNN_F1']:<8.3f} | {m['SVM_Acc']:<8.3f} {m['SVM_F1']:<8.3f} | {m['Binary_Acc']:<8.3f} {m['Binary_F1']:<8.3f} | {m['Subj_Acc']:<8.3f}")
    print("-" * 145)
    
    for name, m in results.items():
        print(f"{name:<20} | {m['MSE']:<8.2f} | {m['KNN_Acc']:<8.3f} {m['KNN_F1']:<8.3f} | {m['SVM_Acc']:<8.3f} {m['SVM_F1']:<8.3f} | {m['Binary_Acc']:<8.3f} {m['Binary_F1']:<8.3f} | {m['Subj_Acc']:<8.3f}")
