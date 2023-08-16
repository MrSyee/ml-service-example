import argparse

import torch
from torchvision import transforms
from PIL import Image

from model import ConvNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST inference Example")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="outputs/mnist_cnn.pt",
        help="model ckpt path",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="torch",
        help="model type. torch | torchscript | onnx",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load model
    if args.model_type == "torchscript":
        model = torch.jit.load(args.ckpt_path)
    else:
        model = ConvNet().to(device)
        model.load_state_dict(torch.load(args.ckpt_path))

    # Prepare input data
    img_path = "ml/mnist_sample.jpg"
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)
    print(img[:,:,:5,:5])

    with torch.no_grad():
        output = model(img)
        print(output)
        pred = output.argmax(dim=1, keepdim=True)

    print(pred)