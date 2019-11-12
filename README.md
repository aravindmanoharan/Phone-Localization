# Phone-Localization
This is an implementation of a simple CNN architecture to locate the phone in an image. The challenge is to train a model with just 130 images. We all knew that deep learning may not perform well with such small dataset. But yet, I used a simple CNN architecture with two convolutional layers and two full connected layers along with some traditional computer vision techiques to get great results as shown in the images below. (The green dot is the predicted center of the phone)

<p align="center">
  <img width="280" alt="SS1" src="https://user-images.githubusercontent.com/35612880/68648670-cc476e80-04d5-11ea-98f9-413bc2ddfccd.png"> <img width="288" alt="SS2" src="https://user-images.githubusercontent.com/35612880/68648676-d4071300-04d5-11ea-8cfb-df40cf6add2e.png"> <img width="280" alt="SS3" src="https://user-images.githubusercontent.com/35612880/68648688-dbc6b780-04d5-11ea-9baf-a21af8d9344e.png">
</p>

## Dataset

This dataset consists of 130 images with one phone in it. Also, it consist of a text file in which it has the coordinates for each of the images. An example image from the dataset can be seen below.

<p align="center">
  <img width="350" src="https://github.com/aravindmanoharan/Phone-Localization/blob/master/find_phone/91.jpg">
</p>

## Simplifying the problem

Since we have just 130 images to train, we can't expect a deep learning model to perform well. Hence I used a sliding window techinque to simplify the problem with a stride of 5. We create a much larger dataset from these images by passing the sliding window of 40 X 40 and labeling if that window contains a phone or not. In this way, we will end up with much larger dataset and the problem becomes a classification problem where given a sliding window the model has to predict if it contains phone or not. 

## Training the model

The CNN architecture used to trained the model contains two convolutional layers and two full connected layer with a binary output. The trained weights are also available in this repository which is used to make the prediction.

## Using the trained model

We can test the model by running the command ```python3 find_phone.py [path/to/test_image]``` which uses the trained weights to make the predictions for each sliding window and then uses traditional computer vision techniques to output the centers of the phone and an image with a green dot (predicted center) like shown in the output images above.

## Have questions? Need help with the code?

If you're having issues with or have questions about the code, [file an issue](https://github.com/aravindmanoharan/Phone-Localization/issues) in this repository so that I can get back to you soon.
