import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

def show_transformed_images(dataset, num_images=5):
    figure = plt.figure(figsize=(20, 4))
    for i in range(num_images):
        # Get a random image
        image, label = dataset[np.random.randint(len(dataset))]
        
        # Add subplot
        ax = figure.add_subplot(1, num_images, i + 1)
        # Convert tensor to numpy and reshape for display
        img_np = image.numpy().squeeze()
        ax.imshow(img_np, cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define the augmentation transforms
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset with transforms
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    
    # Show some transformed images
    show_transformed_images(dataset) 