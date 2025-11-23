import matplotlib.pyplot as plt
import numpy as np

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

def plot_results(results):
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
    plt.show()