import segmentation_models_pytorch as smp
import torch
from dataset import ColorizationDataset
import glob
import onnx


def convert_to_onnx(model_state_dict, inputs):
    """Method to convert pytorch models to onnx format.
    Args:
        model_state_dict: The state dictionary of model.
        inputs: The input tensor
    """
    model = smp.Unet(
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet",
        in_channels=1,
        classes=2,
    )

    model.load_state_dict(model_state_dict)
    model.encoder.set_swish(memory_efficient=False)
    model.eval()
    torch.onnx.export(
        model,
        inputs,
        "generator.onnx",
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


if __name__ == "__main__":
    ckpt = torch.load("./models/model.bin")
    path = "./Dataset"
    paths = glob.glob(path + "/*.jpg")
    sample_ds = ColorizationDataset([paths[0]], img_size=256, split="valid")
    x, y = sample_ds[0]
    batch_size = 1
    inputs = x.unsqueeze(0)
    convert_to_onnx(model_state_dict=ckpt["model_state_dict"]["generator"], inputs=inputs)
    # Check if onnx works
    onnx_model = onnx.load("generator.onnx")
    onnx.checker.check_model(onnx_model)
