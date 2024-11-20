# ML Model CI/CD Pipeline

This repository demonstrates a simple CI/CD pipeline for a Machine Learning project using GitHub Actions. The project includes a Convolutional Neural Network (CNN) model trained on the MNIST dataset, with automated testing and model validation.

## Project Structure
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml
├── model/
│ ├── init.py
│ └── network.py
├── tests/
│ └── test_model.py
├── .gitignore
├── README.md
├── train.py
└── requirements.txt


## Features

- Simple CNN architecture for MNIST digit classification
- Automated model training
- Model validation and testing
- GitHub Actions integration for CI/CD
- CPU-only implementation for better compatibility

## Model Architecture

The model is a simple CNN with:
- 2 convolutional layers
- 2 max-pooling layers
- 2 fully connected layers
- Less than 25,000 parameters
- Input shape: 28x28
- Output: 10 classes (digits 0-9)

## Requirements

- Python 3.8+
- PyTorch (CPU version)
- torchvision
- pytest

## Local Setup

1. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


2. Install dependencies:
bash
pip install -r requirements.txt


3. Train the model:
bash
python train.py



4. Run tests:
bash
pytest tests/


## CI/CD Pipeline

The GitHub Actions workflow (`ml-pipeline.yml`) automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Runs validation tests
5. Archives the trained model

## Tests

The pipeline validates that the model:
- Has less than 25,000 parameters
- Accepts 28x28 input images
- Outputs 10 classes
- Achieves >95% accuracy on test set

## Model Artifacts

Trained models are saved with timestamps and accuracy metrics in the `saved_models/` directory. The naming format is:

    model_timestamp_accuracy.pth


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Notes

- The model is configured for CPU-only training to ensure compatibility with GitHub Actions
- Training is limited to 1 epoch for demonstration purposes
- For production use, consider increasing the number of epochs and adjusting the architecture   