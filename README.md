# EfficientNet-KuzushijiKanji

PyTorch implementation of EfficientNet optimized for Kuzushiji-Kanji character recognition, showcasing state-of-the-art performance on historical Japanese text classification.

![EfficientNet-KuzushijiKanji.png](docs/EfficientNet-KuzushijiKanji.png)

## Table of Contents
- [Background](#background)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [EfficientNet Architecture](#efficientnet-architecture)
- [Contributing](#contributing)
- [License](#license)

## Background

EfficientNet, introduced by Tan and Le in 2019, is a groundbreaking convolutional neural network architecture that achieves state-of-the-art accuracy with an order of magnitude fewer parameters and FLOPS than previous models. It uses a compound scaling method that uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients.

This project adapts EfficientNet for the challenging task of Kuzushiji-Kanji character recognition. Kuzushiji, a cursive writing style used in Japan until the early 20th century, presents unique challenges for optical character recognition due to its complex and varied forms.

## Getting Started

To get started with this project, follow these steps:

Clone the repository:

```bash
git clone https://github.com/yourusername/EfficientNet-KuzushijiKanji.git
cd EfficientNet-KuzushijiKanji
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To use the EfficientNet model in your project, you can utilize the `EfficientNetFactory` class. Here's an example of how to create and use an EfficientNet model:

```python
from efficientnet import EfficientNetFactory
import torch

# Choose the EfficientNet variant you want to use
variant = 'b0'  # Options: 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'
num_classes = 1000  # Adjust based on your Kuzushiji-Kanji dataset

# Create the specified EfficientNet model
model = EfficientNetFactory.create(variant, num_classes)

# Get the appropriate input size for the chosen variant
input_size = EfficientNetFactory.get_input_size(variant)

# Prepare your input data (example)
input_data = torch.randn(1, 3, input_size, input_size)

# Use the model for inference
with torch.no_grad():
 output = model(input_data)

# Process the output as needed
predicted_class = torch.argmax(output, dim=1)
print(f"Predicted class: {predicted_class.item()}")
```

## EfficientNet Architecture

The EfficientNet architecture is built on mobile inverted bottleneck convolution (MBConv) blocks, which were first introduced in MobileNetV2. The network progressively scales up in width, depth, and resolution across different variants (B0 to B7).

For a detailed view of the EfficientNet architecture, refer to our EfficientNet diagram. 

![EfficientNet.png](docs/EfficientNet.png)

## Contributing

We welcome contributions to improve EfficientNet-KuzushijiKanji! If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code adheres to our coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
