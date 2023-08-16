from PIL import Image
import numpy as np

import tritonclient.grpc as grpcclient


def normalize_img(
    img: np.ndarray, mean: float, std: float, eps: float = 1e-7
) -> np.ndarray:
    """Rescale the image in [0, 1] and normalize."""
    img_rescaled = img / (img.max() + eps)
    img_normalized = (img_rescaled - mean) / std
    return img_normalized


if __name__ == "__main__":
    # Set triton client
    triton_client = grpcclient.InferenceServerClient(url=f"localhost:8001")
    model_name = "mnist"
    model_version = "1"

    # Input
    img_path = "ml/mnist_sample.jpg"
    img = Image.open(img_path)
    img = np.array(img)[None, None, :, :]
    img = normalize_img(img, mean=0.1307, std=0.3081)
    img = img.astype("float32")

    # Inference
    input_image = grpcclient.InferInput("INPUT__0", img.shape, "FP32")
    input_image.set_data_from_numpy(img)

    request = triton_client.infer(
        model_name=model_name,
        inputs=[input_image],
        model_version=model_version,
    )
    output = request.as_numpy('OUTPUT__0')
    print(output)
    pred = output.argmax(axis=1)
    print(pred)
