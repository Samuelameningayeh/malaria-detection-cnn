import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_sample_images(loader, class_names, count=5):
    """
    EDA: Plots random images from a batch
    """
    images, labels = next(iter(loader))
    plt.figure(figsize=(15, 5))
    
    for i in range(count):
        plt.subplot(1, count, i + 1)
        # Un-normalize for display: img = (img * std) + mean
        img = images[i].numpy().transpose((1, 2, 0))
        img = img * 0.5 + 0.5
        
        plt.imshow(img)
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.show()

def plot_results(results, save=False):
    """
    Plots Loss and Accuracy curves
    results: dict containing lists of losses and accuracies
    """
    epochs = range(1, len(results['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_loss'], label='Train Loss')
    plt.plot(epochs, results['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_acc'], label='Train Acc')
    plt.plot(epochs, results['val_acc'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    if save:
        plt.savefig('results.png')
    plt.show()

def plot_predictions(model, test_loader, class_names, device='cpu', columns=5, rows=2, save=False):
    """
    Plots predictions for a batch of images.
    model: The trained PyTorch model
    test_loader: DataLoader for test data
    class_names: List of class names
    device: 'cpu' or 'cuda'
    """
    model = model.to(device)
    model.eval() # Important: Turns off Dropout for consistent predictions
    for i in range(count):
        ax = fig.add_subplot(rows, columns, i + 1)
        
        # A. Un-normalize the image so it looks like a photo
        # (Input was normalized with mean=0.5, std=0.5)
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = img * 0.5 + 0.5     # Reverse the math
        img = np.clip(img, 0, 1)  # Ensure colors stay valid
        
        # B. Determine Label Names
        # Use .item() to convert 0-d tensor to scalar
        actual_label = class_names[labels[i].item()]
        pred_label = class_names[preds[i].item()]
        
        # C. Color Code the Title (Green = Correct, Red = Wrong)
        color = 'green' if actual_label == pred_label else 'red'
        
        plt.imshow(img)
        ax.set_title(f"Actual: {actual_label}\nPred: {pred_label}", color=color, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    if save:
        plt.savefig('predictions.png')
    plt.show()