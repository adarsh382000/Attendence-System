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

def data():
    try:
       client = MongoClient(st.secrets["db_address"])
       db = client['Attendence']
       return db
    except Exception:
       st.write("Error connecting to the Database")
       st.stop()

db = data()
train_model = st.secrets["train_model"]

@st.cache(suppress_st_warning=True)
def face_recognition_model():
    try:
        gdown.download(train_model, 'model.h5', quiet=False)
        model = keras.models.load_model('model.h5')
        return model
    except Exception:
        st.write("Error loading predictive model")
        st.stop()

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
    li = list(cursor)
    dist = 1000000.0
    name = "unknown"
    dic = {}
    if len(li) > 0:
        for i in li:
            encdatabase = i['embedding']
            encdatabase = np.array(encdatabase)
            d = cosine(enc,encdatabase)
            dic[i['Name']] = d 
            if d < 0.5 and d < dist:
                name = i['Name']
                dist = d
        return dic
    else:
        return -2
  else:
    return -1


def main():
  st.title("Face Recognition Based Attendence System Prototype")

  activities = ["Mark Attendence", "Admin Login", "Admin Registeration"]
  choice = st.sidebar.selectbox("Menu", activities)

  if choice == "Mark Attendence":
    st.write("**Please use image with frontal angle and image should be well lit**")
    st.write("Example image:")
    st.image('Image.jpg',use_column_width = 'auto')
    uploaded_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
    
    if uploaded_file is not None:
     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
     image = cv2.imdecode(file_bytes, 1)
     if st.button("Proceed"):
         res = test_person(image)
         if type(res) == dict:
           #st.write("Attendence Marked of: ")
           tot = sum(res.values())
           for i,j in res.items():
                my_bar = st.progress(0)
                st.write(i)
                st.write(j/tot)
                st.write(j)
                my_bar.progress((j/tot))
         elif res == -1:
           st.write("No face found, try another image")
         else:
           st.write("Database empty")
    else:
     st.write("Please select an image")
     st.stop()
  
  elif choice == "Admin Login":
    st.write("**Enter your Credentials**")
    userid = st.text_input("UserID(Case-sensitive):")
    password = st.text_input("Password:", type="password")
    
    st.write("Upload the Image of Person to Register them in Database")
    st.write("Example image: ")
    st.image('Image.jpg',use_column_width = 'auto')
    uploaded_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
    
    if uploaded_file is not None:        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
    else:
        st.write("Please Upload an image")
        st.stop()
        
    name = st.text_input("Enter the person's name: ")
    
    if st.button("Proceed"):
        if len(name) == 0:
            st.write("Enter a valid name")
            st.stop()
        cursor = db.Admins.find({'_id' : userid})
        li = list(cursor)
        for i in li:
            pas = i['password']
        if len(li) > 0:
            if pas == password:
                res = add_new_person(name,image)
                if res == 0:
                    st.write("Successfully Registered")
                else:
                    st.write("No face found, try another image")
     
            else:
                st.write("Wrong Password!, try again")

        else:
            st.write('UserID not registered, goto Admin Registration tab')


  elif choice == "Admin Registeration":
    st.write("**Enter the Credentials to register as an Admin**")
    userid = st.text_input("UserID(Case-sensitive):")
    password = st.text_input("Password:", type="password")

    if st.button("Proceed"):
        if len(userid) == 0 or len(password) == 0:
            st.write("Please enter valid credentials")
            st.stop()
        try:
            db.Admins.insert_one({'_id' : userid, 'password' : password})
            st.write("Successfully Registered")
        except Exception:
            st.write("UserID already Exists")

if __name__ == "__main__":
    main()
