# Manga-Colorizer

I like to read manga in my spare time but sometimes the experience is a bit bland since there is no color, hence I decided to put deep learning to good use and train Pix2Pix GAN to colorize manga. 


## Dataset 

For Training data I scraped the images from manga website. I chose one piece since it has huge collection of colorized images. I won't give the scraping scripts nor the dataset. If you want it you can request it to me via mail.


## Training

GAN's are absolutely perfect for such tasks and few years ago a nice paper called [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) which did Image2Image translation. Our task is similar we want to convert B/W to RGB, hence this seems like a nice approach to use for out problem.

I won't go into much detail how the model works and all. I will reference the blogs which I used during the developement.

* [How to Develop a Pix2Pix GAN for Image-to-Image Translation](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/)
* [A Gentle Introduction to Pix2Pix Generative Adversarial Network](https://machinelearningmastery.com/a-gentle-introduction-to-pix2pix-generative-adversarial-network/)
* [Pix2Pix](https://towardsdatascience.com/pix2pix-869c17900998)
* [Pix2Pix Implementation from Scratch](https://www.youtube.com/watch?v=SuddDSqGRzg&ab_channel=AladdinPersson)

The only thing I observed during my experiments was it took time to converge when generator was trained from scratch. So here is what is I did.

I used [segmentation-models](https://pypi.org/project/segmentation-models/) library to create a Unet architecture with efficientnet-b1 as the backbone. Then I subsampled some of the images from dataset and pretrained the Unet using just L1-Loss. 

Finally I used the pretrained Unet along with discriminator and trained both of them adversarial fashion. This gave me good results but I am limited on GPU compute and did not train the model for too long due to these constraints.

For handling most of the boiler plate code, I used [TorchFlare](https://github.com/Atharva-Phatak/torchflare) because of its easy to use API and customization to callabacks. 

Thats it, the only time I spend was training the models everything else was handled by TorchFlare.


## Streamlit APP and ONNX conversion

I first converted the pytorch model to onnx format so that we can use it with any framework. I have also created a small streamlit based application which let's you colorize the images.

I have provided the scripts for onnx model conversion and the streamlit app.

![](https://raw.githubusercontent.com/Atharva-Phatak/Manga-Colorizer/main/outputs/streamlit.png)

## Results

Since it was only trained on one-piece manga images it may or may not perform good on other kinds of manga. But for now let's see some one piece images it created. Also the images are of just 256x256 resolution due to limitations on my hardware.

| Orginal Colored Version | Generated Colored Version |
|-------------------------|---------------------------|
|![](https://raw.githubusercontent.com/Atharva-Phatak/Manga-Colorizer/main/outputs/real_images_1.jpg) | ![](https://raw.githubusercontent.com/Atharva-Phatak/Manga-Colorizer/main/outputs/fake_images_1.jpg)|
|![](https://raw.githubusercontent.com/Atharva-Phatak/Manga-Colorizer/main/outputs/real_images_2.jpg) | ![](https://raw.githubusercontent.com/Atharva-Phatak/Manga-Colorizer/main/outputs/fake_images_2.jpg)|
|![](https://raw.githubusercontent.com/Atharva-Phatak/Manga-Colorizer/main/outputs/real_images_3.jpg) | ![](https://raw.githubusercontent.com/Atharva-Phatak/Manga-Colorizer/main/outputs/fake_images_3.jpg)|


## Future Work

* I plan on developing this in my free time and I am going to work more on dataset curation and model experimentation with new models. 
* If you would like to contribute please open up issue with the idea and we can discuss more.
