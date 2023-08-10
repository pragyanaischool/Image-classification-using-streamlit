# Image-classification-using-streamlit
This is an image classification web application deployed using Streamlit

## Requirements
!pip install streamlit opencv-python tensorflow
IDE of your choice: VS Code, Google Colab, Kaggle notebook

## Contents
The "data" folder contains the images train set  
The "newtest" folder contains the images test set  

## Steps
1) Create the model from Google Teachable Machine by uploading the images and train the model. You can find it through this link:
https://teachablemachine.withgoogle.com/

2) Export and Download the model as Tensorflow NOT Tensorflow.lite, Tensorflow.js. Extract the contents from the zip folder.

3) Create a python script "ïmage.py" and put the Streamlit code. Ensure the keras_model.h5, labels.txt and the image.py are in the same folder

4) Run the code: streamlit run image.py