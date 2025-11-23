import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision 

def plot_sample_images(dataloader, class_names, num_images=8):
    """
    Plots a sample of images from a dataloader.
    """
    images, labels = next(iter(dataloader))
    fig = plt.figure(figsize=(num_images * 2, 4))
    for i in range(min(num_images, len(images))): # Ensure we don't try to plot more than available
        ax = fig.add_subplot(2, num_images // 2, i + 1)
        img = images[i].cpu().numpy().transpose((1, 2, 0))

        # Denormalize if necessary (assuming standard ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.set_title(class_names[labels[i]])
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def plot_results(results, save=False):
    """
    Plots training and validation loss and accuracy curves.
    """
    epochs = range(1, len(results['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_loss'], label='Train Loss')
    plt.plot(epochs, results['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_acc'], label='Train Accuracy')
    plt.plot(epochs, results['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save:
        plt.savefig("train_results.png")
    plt.show()

def plot_predictions(model, test_loader, class_names, device, columns=4, rows=2, save=False):
    """
    Plots a grid of sample images with their true and predicted labels.
    """
    model = model.to(device)
    model.eval() # Important: Turns off Dropout for consistent predictions
    fig = plt.figure(figsize=(columns * 3, rows * 4)) # Adjust figure size

    images, labels = next(iter(test_loader)) # Get one batch for predictions
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

    # Plot up to columns*rows images from the batch
    for i in range(min(len(images), columns * rows)):
        ax = fig.add_subplot(rows, columns, i + 1) # Use i as the subplot index

        # Denormalize image (assuming standard ImageNet normalization as used in dataloading.py usually)
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406]) # Standard ImageNet mean
        std = np.array([0.229, 0.224, 0.225])   # Standard ImageNet std
        img = std * img + mean
        img = np.clip(img, 0, 1) # Clip to [0, 1] range for imshow

        ax.imshow(img)
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        title_color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=title_color, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if save:
        plt.savefig("predictions.png")
    plt.show()