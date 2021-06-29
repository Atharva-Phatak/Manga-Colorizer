import onnxruntime
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
from torchvision.utils import save_image


ort_session = onnxruntime.InferenceSession("generator.onnx")


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    images = np.stack(rgb_imgs, axis=0)
    return torch.from_numpy(images).permute(0, 3, 1, 2)


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def get_input_tensor(image_path):

    infer_transforms = transforms.Resize((256, 256))
    img = Image.open(image_path).convert("RGB")
    img = infer_transforms(img)
    org_img = transforms.ToTensor()(img)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50.0 - 1.0
    L = L.unsqueeze(0)
    return L, org_img


def do_inference(image_path):

    L = get_input_tensor(image_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(L)}
    ort_output = ort_session.run(None, ort_inputs)
    fake_ab = ort_output[0]
    fake_ab = torch.from_numpy(fake_ab)
    rgb_img = lab_to_rgb(L, fake_ab)

    save_image(rgb_img, "new_img.png", nrow=1)


if __name__ == "__main__":
    image_path = "/home/atharvap/Desktop/manga-colorize/my_hero_academia_310_3.jpg"
    do_inference(image_path)
