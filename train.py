import torch
from tqdm.auto import tqdm # Automatically chooses notebook-friendly progress bars

def train_step(model, loader, criterion, optimizer, device):
    """
    Runs one epoch of training. Returns average loss and accuracy.
    """
    model.train() # Set to train mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    # leave=False means the bar disappears after the epoch finishes (keeps notebook clean)
    loop = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # UPDATE THE BAR LIVE
        loop.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc