import streamlit as st
import onnxruntime
from torchvision import transforms
from model_inference import lab_to_rgb, get_input_tensor, to_numpy
import torch

ort_session = onnxruntime.InferenceSession("generator.onnx")

st.title("Colorize One-Piece Manga using Pix2Pix GAN.")
st.write(
    "I am a anime fan but the episodes are so slow to release I usually end up reading manga but they are not colored. \
    Hence I decided to put Deep Learning to good use and trained a Generative Adversarial network to colorize Manga."
)
st.write(
    "A fair warning, I am limited on compute power hence I cannot train high-res GANs or big models. But you can use code for the training and do a much better work if you have Good GPU's"
)
st.markdown(
    "Do give a :stars: if you like the work.  \
    Github: https://github.com/Atharva-Phatak/Manga-Colorizer"
)
file_up = st.file_uploader("Upload an image", type=["jpg", "png"])


def do_inference(image_path):

    L, org_img = get_input_tensor(image_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(L)}
    ort_output = ort_session.run(None, ort_inputs)
    fake_ab = ort_output[0]
    fake_ab = torch.from_numpy(fake_ab)
    rgb_img = lab_to_rgb(L, fake_ab)
    img = transforms.ToPILImage()(rgb_img.squeeze(0))
    org_img = transforms.ToPILImage()(org_img.squeeze(0))
    st.image(
        [org_img, img],
        caption=["Original Image", "Generated Image"],
        use_column_width=False,
    )


if file_up is None:
    print("Please upload a file")
else:
    do_inference(file_up)
