from data_loader import GaitDataLoader

# Load GEI data for CNN training
loaders = GaitDataLoader.create_loaders(
    data_root='output',
    data_type='gei',  # Use GEI images
    batch_size=32
)

# Training loop
for gei_images, labels, metadata in loaders['train']:
    # gei_images shape: (32, 1, 128, 64)
    # labels: subject IDs
    # Train your CNN model here
    pass
```

---

## 📝 Important Notes:

1. **Pose data won't work** with CASIA-B (it's silhouettes, not RGB), so use with GEI/silhouette-based models

2. **Your `output/` folder structure:**
```
   output/
   ├── 001/ ... 005/  (subject folders with .npy files)
   ├── data_splits.json  (train/val/test splits)
   └── dataset_statistics.json  (dataset info)