cat > usage_example.md << 'EOF'
# Usage Example for Model Training

## Loading Preprocessed Data
```python
from data_loader import GaitDataLoader

# Load GEI data for CNN training
loaders = GaitDataLoader.create_loaders(
    data_root='output',
    data_type='gei',  # Use GEI images (recommended!)
    batch_size=32
)

# Get dataset info
print(f"Training batches: {len(loaders['train'])}")
print(f"Validation batches: {len(loaders['val'])}")
print(f"Test batches: {len(loaders['test'])}")

# Training loop example
for epoch in range(10):
    for gei_images, labels, metadata in loaders['train']:
        # gei_images shape: (batch_size, 1, 128, 64)
        # labels: subject IDs as integers (0, 1, 2, 3, 4 for our 5 subjects)
        # metadata: dict with 'subject_id', 'sequence_id', 'view_angle'
        
        # YOUR MODEL TRAINING CODE HERE
        # outputs = model(gei_images)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
        pass
```

## Alternative: Load Silhouette Sequences (for RNN/LSTM/3D-CNN)
```python
loaders = GaitDataLoader.create_loaders(
    data_root='output',
    data_type='silhouettes',  # Temporal sequences
    batch_size=16  # Smaller batch size due to memory
)

for silhouette_sequences, labels, metadata in loaders['train']:
    # silhouette_sequences shape: (batch_size, num_frames, 1, 128, 64)
    # Feed to 3D CNN or reshape for LSTM
    pass
```

## Check Dataset Info
```python
from data_loader import get_dataset_info
import json

info = get_dataset_info('output')
print(json.dumps(info, indent=2))
```

## Visualize a Sample
```python
from visualization import visualize_sample

visualize_sample(
    data_root='output',
    subject_id='001',
    sequence_id='nm-01',
    view_angle='090',
    save_dir='viz'
)
```

## Important Notes

1. **Pose data won't work** with CASIA-B (silhouettes only, not RGB). Use GEI or silhouette-based models.
2. **Data is ready** - train/val/test splits already created
3. **Start with GEI + simple CNN** for fastest results
EOF