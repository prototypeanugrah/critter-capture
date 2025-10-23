import torch

from critter_capture.models import AnimalSpeciesCNN


def test_model_forward_shape() -> None:
    model = AnimalSpeciesCNN(
        in_channels=3,
        num_classes=6,
        conv_filters=[16, 32, 64, 128, 128],
        hidden_dim=128,
        dropout=0.3,
    )

    inputs = torch.randn(2, 3, 224, 224)
    outputs = model(inputs)
    assert outputs.shape == (2, 6)
