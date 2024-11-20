import torch
import pytest
import os
from model.network import SimpleCNN
from torchvision import datasets, transforms
from train import train

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = SimpleCNN()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_input_shape():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
        assert True
    except:
        assert False, "Model failed to process 28x28 input"

def test_output_shape():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape[1] == 10, f"Output shape is {output.shape[1]}, should be 10"

def test_model_accuracy():
    # First ensure we have a trained model
    if not os.path.exists('saved_models') or not os.listdir('saved_models'):
        print("No trained model found. Training a new model...")
        train()
    
    model = SimpleCNN()
    
    # Load the latest model
    import glob
    model_files = glob.glob('saved_models/model_*.pth')
    assert len(model_files) > 0, "No model files found even after training"
    
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
    
    # Test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Accuracy is {accuracy}%, should be > 95%" 