import argparse

import onnxruntime as ort
import onnx
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch MNIST inference Example")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="outputs/onnx/mnist_cnn.onnx",
        help="model ckpt path",
    )
    return parser.parse_args()


def normalize_img(
    img: np.ndarray, mean: float, std: float, eps: float = 1e-7
) -> np.ndarray:
    """Rescale the image in [0, 1] and normalize."""
    img_rescaled = img / (img.max() + eps)
    img_normalized = (img_rescaled - mean) / std
    return img_normalized


if __name__ == "__main__":
    args = parse_args()

    # Check validation onnx model
    onnx_model = onnx.load(args.ckpt_path)
    onnx.checker.check_model(onnx_model)

    # Load image
    img_path = "ml/mnist_sample.jpg"
    img = Image.open(img_path)
    img = np.array(img)[None, None, :, :].astype("float32")
    img = normalize_img(img, mean=0.1307, std=0.3081)
    print(img[:,:,:5,:5])

    # Inference
    ort_session = ort.InferenceSession(args.ckpt_path)
    input_name = ort_session.get_inputs()[0].name
    print(input_name)

    output = ort_session.run(None, {input_name: img})[0]
    print(type(output), output)
    pred = output.argmax(axis=1)
    print(pred)