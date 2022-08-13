Dataset from 
https://www.kaggle.com/datasets/chaitanyasapre/cats-and-dogs-dataset
Model in h5 Extension on
https://drive.google.com/drive/u/0/folders/1Yyqnx6-kdLDRqJpLfUIhNlkyRqRzNoH9
---

for running the model 
---
Download the Model.h5 from Google Drive
---
If you don't have a python environment then Open Google colab or Jupyter then Upload the Model.h5 file there 
---
Now enter this code
---
---
---
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
a = tf.keras.models.load_model("Model.h5")
import cv2
def is_this_a_cat_or_a_dog(path):
    #reading the image from path
    im = cv2.imread(path)
    #reshaping image for inputting in tensor flow
    reshaped_im =  tf.reshape(cv2.resize(im,(200,200)),[1,200,200,3])
    Var = a.predict(reshaped_im)
    print(Var)
    #Checking if it's resemble to cat or not
    if Var >= 0.5:
        print("It's a Cat")
    else:
        print("It's a dog")
is_this_a_cat_or_a_dog("Your Image")
plt.imshow(cv2.imread("Your Image"))
---
---
or simply run
Try_model_here.ipynb
