import os

import torch

from model import ConvNet

batch_size = 1
ckpt_path = "outputs/mnist_cnn.pt"
onnx_path = "outputs/onnx"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load model
    model = ConvNet()
    model.load_state_dict(torch.load(ckpt_path))

    # Convert ONNX
    if not os.path.exists(onnx_path):
        os.mkdir(onnx_path)

    dummy_input = torch.randn(batch_size, 1, 28, 28, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        os.path.join(onnx_path, "mnist_cnn.onnx"),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}},
    )
