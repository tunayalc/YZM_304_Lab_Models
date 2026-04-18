import torch

from src.models import HybridFeatureCNN, LeNetBaselineCNN, LeNetImprovedCNN, ResNet18ReferenceCNN


def test_lenet_baseline_output_shape() -> None:
    model = LeNetBaselineCNN(num_classes=10)
    logits = model(torch.randn(4, 1, 32, 32))
    assert logits.shape == (4, 10)


def test_lenet_improved_output_shape() -> None:
    model = LeNetImprovedCNN(num_classes=10)
    logits = model(torch.randn(4, 1, 32, 32))
    assert logits.shape == (4, 10)


def test_resnet18_reference_output_shape() -> None:
    model = ResNet18ReferenceCNN(num_classes=10)
    logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 10)


def test_hybrid_feature_cnn_extract_features_shape() -> None:
    model = HybridFeatureCNN(num_classes=10)
    features = model.extract_features(torch.randn(3, 1, 32, 32))
    assert features.shape == (3, 256)
