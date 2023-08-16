import os

import torch

from model import ConvNet

batch_size = 1
ckpt_path = "outputs/mnist_cnn.pt"
torchscript_path = "outputs/torchscript"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load model
    model = ConvNet()
    model.load_state_dict(torch.load(ckpt_path))

    # Convert ONNX
    if not os.path.exists(torchscript_path):
        os.mkdir(torchscript_path)

    dummy_input = torch.randn(batch_size, 1, 28, 28, device=device)
    traced_model = torch.jit.trace(model, (dummy_input), strict=False)
    traced_model.save(f"{torchscript_path}/mnist_cnn.pt")
