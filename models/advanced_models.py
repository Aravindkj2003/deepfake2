import torch
import torch.nn as nn
from torchvision import models


def _adapt_resnet_first_conv(conv_layer: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:] = conv_layer.weight.mean(dim=1, keepdim=True)
        if conv_layer.bias is not None:
            new_conv.bias[:] = conv_layer.bias
    return new_conv


def _adapt_efficientnet_first_conv(conv_layer: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv_layer.out_channels,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:] = conv_layer.weight.mean(dim=1, keepdim=True)
        if conv_layer.bias is not None:
            new_conv.bias[:] = conv_layer.bias
    return new_conv


def create_advanced_model(model_name: str, pretrained: bool = True) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.conv1 = _adapt_resnet_first_conv(model.conv1)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.conv1 = _adapt_resnet_first_conv(model.conv1)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        first_conv = model.features[0][0]
        model.features[0][0] = _adapt_efficientnet_first_conv(first_conv)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        return model

    raise ValueError(f"Unsupported advanced model: {model_name}")
