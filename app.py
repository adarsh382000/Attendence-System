from tensorflow import keras
import mtcnn
import cv2
import numpy as np
import os
import random
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
from pymongo import MongoClient
import pymongo
import streamlit as st
import gdown

client = MongoClient(st.secrets["db_address"])
db = client['Attendence']

@st.cache
def face_recognition_model():
    try:
        gdown.download('https://drive.google.com/uc?id=1oOqvp0xR01oW_1jLnzY5jubkf7vR0B29', 'model.h5', quiet=False)
        model = keras.models.load_model('model.h5')
        return model
    except Exception:
        st.write("Error loading predictive model")

model = face_recognition_model()

def detect_face(ad):
  detector = mtcnn.MTCNN()
  return detector.detect_faces(ad)

def get_emb(face):
  face = normalize(face)
  face = cv2.resize(face,(160,160))
  return model.predict(np.expand_dims(face, axis=0))[0]

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def add_new_person(name,img):
  l2 = Normalizer()
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  res = detect_face(img)
  if res:
    res = max(res, key = lambda b: b['box'][2] *b['box'][3])
    img, _, _ = get_face(img,res['box'])
    enc = get_emb(img)
    enc = l2.transform(enc.reshape(1,-1))[0]
    enc = enc.tolist()
    db.embd.insert_one({'Name' : name, 'embedding' : enc})
    return 0
  else:
    return -1

def test_person(img):
  new_enc = {}
  l2 = Normalizer()
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  res = detect_face(img)
  if res:
    res = max(res, key = lambda b: b['box'][2] *b['box'][3])
    img,_,_ = get_face(img,res['box'])
    enc = get_emb(img)
    enc = l2.transform(enc.reshape(1,-1))[0]
    name  = 'unknown'
    dist = float('inf')
    cursor = db.embd.find()
    if len(list(cursor)) > 0:
        for i in cursor:
            encdatabase = i['embedding']
            name = i['Name']
            encdatabase = np.array(encdatabase)
            d = cosine(enc,encdatabase)
            new_enc[name] = d
            return new_enc
    else:
        return -2
  else:
    return -1

def main():
  st.title("Face Recognition Based Attendence System Prototype")
  st.write("**Using FaceNet and MongoDB**")
  

  activities = ["Admin Login", "Mark Attendence", "Admin Registeration"]
  choice = st.sidebar.selectbox("Menu", activities)

  if choice == "Mark Attendence":
    st.write("Please use image with frontal angle and image should be well lit")
    uploaded_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
    
    if uploaded_file is not None:
      file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
      image = cv2.imdecode(file_bytes, 1)
      if st.button("Proceed"):
         res = test_person(image)
         if res == -2:
           st.write("Database empty")
         elif res == -1:
           st.write("No face found, try another image")
         else:
           st.write("Attendence Marked")
           st.image(image, use_column_width = True)
  
  elif choice == "Admin Login":
    st.write("Enter your Credentials")
    userid = st.text_input("UserID: ")
    password = st.text_input("Password: ", type="password")
    
    st.write("Upload the Image of Person to Register them in DataBase")
    uploaded_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

    if st.button("Proceed"):
        cursor = db.Admins.find({'_id' : userid})
        li = list(cursor)
        for i in li:
            pas = i['password']
        if len(li) > 0:
            if str(pas) == str(password):
                name = st.text_input("Enter the person's name: ")
                if st.button("Register"):
                    res = add_new_person(name,image)
                    if res == 0:
                        st.write("Successfully Registered")
                    else:
                        st.write("No face found, try another image")

            else:
                st.write("Wrong Password!, try again")

        else:
            st.write('UserID not registered, goto Admin Regesitration tab')


  elif choice == "Admin Registeration":
    st.write("Enter the Credentials to register as an Admin")
    userid = st.text_input("UserID: ")
    password = st.text_input("Password: ", type="password")

    if st.button("Proceed"):
        try:
            db.Admins.insert_one({'_id' : userid, 'password' : password})
            st.write("Successfully Registered")
        except Exception:
            st.write("UserID already Exists")

if __name__ == "__main__":
    main()
