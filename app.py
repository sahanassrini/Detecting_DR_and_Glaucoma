import streamlit as st
import json
import requests
import base64
from PIL import Image
import io

#CONSTANTS
PREDICTED_LABELS = ['Normal','Glaucoma','Diabetic retinopathy' ]
IMAGE_URL = "https://www.aha.org/sites/default/files/inline-images/Nebraska-Medicine-Targets-Earlier-Diabetic-Retinopathy-Detection-with-AI.png"
PREDICTED_LABELS.sort()

def get_prediction(image_data):
  #replace your image classification ai service endpoint URL
  url = 'https://askai.aiclub.world/be36ec12-1137-42d1-9023-4d8e0a8200c5'  
  r = requests.post(url, data=image_data)
  response = r.json()['predicted_label']
  score = r.json()['score']
  #print("Predicted_label: {} and confidence_score: {}".format(response,score))
  return response, score



#Building the website

#title of the web page
st.title("Eye Condition Classification")

#setting the main picture
st.image(IMAGE_URL, caption = "Image Classification")

#about the web app
st.header("About the Web App")

#details about the project
with st.expander("Web App 🌐"):
    st.subheader("Glaucoma and Diabetic retinopathy Classification")
    st.write("""My web app is designed to classify and predict eye conditions based on input image. It identifies three categories:
    1.Normal
    2.Glaucoma
    3.Diabetic Retinopathy
    """)

#setting file uploader
image =st.file_uploader("Upload an image",type = ['jpg','png','jpeg'])
if image:
  #converting the image to bytes
  img = Image.open(image).convert('RGB') #ensuring to convert into RGB as model expects the image to be in 3 channel
  buf = io.BytesIO()
  img.save(buf,format = 'JPEG')
  byte_im = buf.getvalue()

  #converting bytes to b64encoding
  payload = base64.b64encode(byte_im)

  #file details
  file_details = {
    "file name": image.name,
    "file type": image.type,
    "file size": image.size
  }

  #write file details
  #st.write(file_details) #uncomment if you need to show file details

  #setting up the image
  st.image(img)

  #predictions
  response, scores = get_prediction(payload)

  #if you are using the model deployment in navigator
  #you need to define the labels
  response_label = PREDICTED_LABELS[response]

  st.metric("Prediction Label",response_label)
  st.metric("Confidence Score", max(scores))
  
