import torch.nn as nn


class MBConv(nn.Module):
    """
    MBConv (Mobile Inverted Residual Bottleneck) block for EfficientNet.

    This block consists of an expansion phase, a depthwise convolution, and a projection phase.
    It may include a skip connection if input and output dimensions match.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for the depthwise convolution.
        stride (int): Stride for the depthwise convolution.
        expand_ratio (int): Expansion ratio for the expand phase.

    Attributes:
        stride (int): Stride of the depthwise convolution.
        expand_ratio (int): Expansion ratio for the expand phase.
        expand_channels (int): Number of channels after expansion.
        skip_connection (bool): Whether to use skip connection.

    Note:
        If expand_ratio is 1, the expansion phase is skipped.
        Skip connection is used when stride is 1 and input channels equal output channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.expand_channels = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels,
                                         self.expand_channels,
                                         kernel_size=1,
                                         bias=False)
            self.expand_bn = nn.BatchNorm2d(self.expand_channels)
            self.expand_act = nn.SiLU()

        self.depthwise_conv = nn.Conv2d(self.expand_channels,
                                        self.expand_channels,
                                        kernel_size,
                                        stride,
                                        padding=kernel_size // 2,
                                        groups=self.expand_channels,
                                        bias=False)
        self.depthwise_bn = nn.BatchNorm2d(self.expand_channels)
        self.depthwise_act = nn.SiLU()

        self.project_conv = nn.Conv2d(self.expand_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

        self.skip_connection = stride == 1 and in_channels == out_channels

    def forward(self, x):
        """
        Forward pass of the MBConv block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the MBConv block.
        """
        identity = x

        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_act(x)

        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_act(x)

        x = self.project_conv(x)
        x = self.project_bn(x)

        if self.skip_connection:
            x = x + identity

        return x


class EfficientNet(nn.Module):
    """
    EfficientNet: A scalable convolutional neural network architecture.

    This class implements the EfficientNet architecture as described in
    "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
    by Tan and Le (2019).

    Args:
        width_coefficient (float): Scaling coefficient for network width.
        depth_coefficient (float): Scaling coefficient for network depth.
        dropout_rate (float): Dropout rate for the final layer.
        num_classes (int, optional): Number of output classes. Defaults to 1000.

    Attributes:
        stem_conv (nn.Conv2d): Initial convolutional layer.
        stem_bn (nn.BatchNorm2d): Batch normalization for the stem.
        stem_act (nn.SiLU): Activation function for the stem.
        blocks (nn.ModuleList): List of MBConv blocks.
        head_conv (nn.Conv2d): Final 1x1 convolutional layer.
        head_bn (nn.BatchNorm2d): Batch normalization for the head.
        head_act (nn.SiLU): Activation function for the head.
        avg_pool (nn.AdaptiveAvgPool2d): Global average pooling.
        dropout (nn.Dropout): Dropout layer.
        fc (nn.Linear): Final fully connected layer.

    Note:
        The network architecture is dynamically adjusted based on the
        width and depth coefficients. The base configuration is scaled
        according to these coefficients to create different EfficientNet variants.
    """

    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, num_classes=2):
        super(EfficientNet, self).__init__()

        # Base configuration
        base_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        base_depths = [1, 2, 2, 3, 3, 4, 1]
        base_strides = [1, 2, 2, 2, 1, 2, 1]
        base_expand_ratios = [1, 6, 6, 6, 6, 6, 6]

        # Scale the channels and depths based on the coefficients
        channels = [int(c * width_coefficient) for c in base_channels]
        depths = [int(d * depth_coefficient) for d in base_depths]

        # Initial layers
        self.stem_conv = nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels[0])
        self.stem_act = nn.SiLU()

        # MBConv blocks
        self.blocks = nn.ModuleList()
        for i in range(7):
            for j in range(depths[i]):
                stride = base_strides[i] if j == 0 else 1
                in_channels = channels[i] if j == 0 else channels[i+1]
                out_channels = channels[i+1]
                kernel_size = 3 if i == 0 else 5
                expand_ratio = base_expand_ratios[i]
                self.blocks.append(MBConv(in_channels, out_channels, kernel_size, stride, expand_ratio))

        # Final layers
        self.head_conv = nn.Conv2d(channels[-2], channels[-1], kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(channels[-1])
        self.head_act = nn.SiLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        """
        Forward pass of the EfficientNet model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_act(x)

        for block in self.blocks:
            x = block(x)

        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.head_act(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class EfficientNetConfig:
    """
    Configuration class for EfficientNet variants.

    This class encapsulates the key parameters that define different variants
    of the EfficientNet model architecture.

    Attributes:
        width_coefficient (float): Scaling coefficient for network width.
            Controls the number of channels in each layer.
        depth_coefficient (float): Scaling coefficient for network depth.
            Controls the number of layers in the network.
        resolution (int): Input resolution (height and width) for the model.
            Determines the size of input images the model expects.
        dropout_rate (float): Dropout rate applied in the final layers.
            Used for regularization to prevent overfitting.

    These parameters are used to create different variants of EfficientNet (B0-B7)
    by scaling the baseline architecture according to the compound scaling method
    described in the EfficientNet paper.

    Example:
        config = EfficientNetConfig(1.0, 1.0, 224, 0.2)  # EfficientNet-B0 configuration
    """

    def __init__(self, width_coefficient, depth_coefficient, resolution, dropout_rate):
        """
        Initialize an EfficientNetConfig instance.

        Args:
            width_coefficient (float): Scaling factor for network width.
            depth_coefficient (float): Scaling factor for network depth.
            resolution (int): Input image resolution.
            dropout_rate (float): Dropout rate for regularization.
        """
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.resolution = resolution
        self.dropout_rate = dropout_rate


class EfficientNetFactory:
    """
    Factory class for creating different variants of EfficientNet models.

    This class provides static methods to create EfficientNet models of various
    sizes (B0 to B7) and to retrieve the appropriate input size for each variant.

    The factory encapsulates the configuration details for each EfficientNet variant,
    making it easy to instantiate specific models without needing to remember the
    exact scaling coefficients and parameters.

    Methods:
        create(variant, num_classes=1000):
            Creates and returns an EfficientNet model of the specified variant.

        get_input_size(variant):
            Returns the recommended input image size for the specified variant.

    Usage:
        model = EfficientNetFactory.create('b0', num_classes=1000)
        input_size = EfficientNetFactory.get_input_size('b0')
    """

    @staticmethod
    def create(variant, num_classes=1000):
        """
        Create and return an EfficientNet model of the specified variant.

        Args:
            variant (str): The EfficientNet variant to create ('b0', 'b1', ..., 'b7').
            num_classes (int, optional): Number of classes for the model's output layer.
                                         Defaults to 1000.

        Returns:
            EfficientNet: An instance of the specified EfficientNet variant.

        Raises:
            ValueError: If an unknown variant is specified.

        Example:
            model = EfficientNetFactory.create('b0', num_classes=100)
        """

        configs = {
            'b0': EfficientNetConfig(1.0, 1.0, 224, 0.2),
            'b1': EfficientNetConfig(1.0, 1.1, 240, 0.2),
            'b2': EfficientNetConfig(1.1, 1.2, 260, 0.3),
            'b3': EfficientNetConfig(1.2, 1.4, 300, 0.3),
            'b4': EfficientNetConfig(1.4, 1.8, 380, 0.4),
            'b5': EfficientNetConfig(1.6, 2.2, 456, 0.4),
            'b6': EfficientNetConfig(1.8, 2.6, 528, 0.5),
            'b7': EfficientNetConfig(2.0, 3.1, 600, 0.5),
        }

        if variant not in configs:
            raise ValueError(f"Unknown EfficientNet variant: {variant}")

        config = configs[variant]
        return EfficientNet(config.width_coefficient, config.depth_coefficient,
                            config.dropout_rate, num_classes)

    @staticmethod
    def get_input_size(variant):
        """
        Get the recommended input image size for the specified EfficientNet variant.

        Args:
            variant (str): The EfficientNet variant ('b0', 'b1', ..., 'b7').

        Returns:
            int: The recommended input image size (both height and width) for the variant.
                 Returns 224 as default if the variant is not recognized.

        Example:
            input_size = EfficientNetFactory.get_input_size('b3')
        """
        configs = {
            'b0': 224, 'b1': 240, 'b2': 260, 'b3': 300,
            'b4': 380, 'b5': 456, 'b6': 528, 'b7': 600
        }
        return configs.get(variant, 224)  # Default to 224 if variant not found


if __name__ == "__main__":
    import torch
    from torchviz import make_dot


    # Get the appropriate input size for the chosen variant
    resolution = EfficientNetFactory.get_input_size(variant)

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, resolution, resolution)

    # Generate the dot graph
    dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))

    # Render and save the graph
    dot.render(f'EfficientNet', format='png', cleanup=True)
    print(f"Model visualization has been saved as 'EfficientNet.png'")

    # Print model summary and total parameters
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
