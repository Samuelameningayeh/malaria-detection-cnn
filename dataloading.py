import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def create_dataloaders(data_dir, batch_size=32, img_size=128):
    """
    Returns train_loader, val_loader, and class_names
    """
    # 1. Define Transformations
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(), # Data Augmentation
        transforms.RandomRotation(10),     # Data Augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 2. Load Data
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    class_names = full_dataset.classes # ['Parasitized', 'Uninfected']
    
    # 3. Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # 4. Create Loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, class_names