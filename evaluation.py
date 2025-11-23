import torch
from tqdm.auto import tqdm

def evaluate_step(model, loader, criterion, device):
    """
    Runs validation. Returns average loss and accuracy.
    """
    model.eval() # Set to eval mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    # WRAP THE LOADER
    loop = tqdm(loader, desc="Validating", leave=False)
    
    with torch.no_grad(): # No gradients needed
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc