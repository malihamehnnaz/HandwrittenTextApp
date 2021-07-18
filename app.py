import cv2
import pathlib
import dropbox
import re
from PIL import *
import glob
import numpy as np
import pandas as pd
import os
from statistics import mean
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
from tensorflow.keras.models import load_model
import pickle
from pickle import dump
from fpdf import FPDF
from PIL import Image, ImageOps
from flask import Flask,render_template,request,send_file
import base64
from PIL import Image as Img
import imutils
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from pathlib import Path
from email import encoders

global kkk
global model
receiver_address=""
app = Flask(__name__)

model =load_model("all_model.h5")
LB = LabelBinarizer()


@app.route('/',methods=["GET","POST"])
def index():
    global receiver_address
    if request.method == "POST":
        receiver_address=str(request.form.get("name"))
        print(receiver_address)
    return render_template('index.html')
        

def get_letters(img):
    letters = []
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    
    for c in cnts:
        if cv2.contourArea(c) > 3000:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x+ w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = thresh.reshape(1,32,32,1)
            ypred = model.predict(thresh)
            ypred = LB.inverse_transform(ypred)
            [x] = ypred
            letters.append(x)
    return letters, image


def get_word(letter):
    word = "".join(letter)
    return word

def mainpdf():
    with open(r"someobjectE.pickle", "rb") as input_file:
        e = pickle.load(input_file)
    with open(r"someobjectF.pickle", "rb") as input_file:
        f = pickle.load(input_file)
    train_Y = LB.fit_transform(e)
    val_Y = LB.fit_transform(f)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=15)

    dir = "LineOutput"
    sentence = []
    s = " "
    k = 0
    for i in os.listdir(dir):
        list = os.listdir(dir) # dir is your directory path
        for j in os.listdir(dir):
            letter,image = get_letters(os.path.join(dir,j))
            word= get_word(letter)
            sentence.insert(k,word) 
            k=k+1
        s=s.join(sentence) 
        n=40
        b=([s[i:i+n] for i in range(0, len(s), n)])
        l=0
        a=1
        while l<len(b):
            m=(b[l])
            pdf.cell(200, 10, txt =m, ln=a, align = 'L')
            a=a+1
            l=l+1
        pdf.output("GFG.pdf") 
    print(sentence)
    
    
    mail_content = '''Mail Received from HandWritten Text Recognition'''
    
    sender_address = "testing1122335@gmail.com"
    sender_pass = "maliha_05"
    global receiver_address
    print(receiver_address)
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'A test mail sent by Python. It has an attachment.'
    message.attach(MIMEText(mail_content, 'plain'))
    attach_file_name = 'GFG.pdf'
    attach_file = open(attach_file_name, 'rb')
    payload = MIMEBase('application', 'octate-stream')
    payload.set_payload((attach_file).read())
    encoders.encode_base64(payload)
    payload.add_header('Content-Decomposition', 'attachment; filename="{}"'.format(Path(attach_file_name).name))
    message.attach(payload)
    session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
    session.starttls()  # enable security
    session.login(sender_address, sender_pass)  # login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    filename = "GFG.pdf"
    folder = pathlib.Path("C:/Users/HP/Desktop/Final Project/Flask")   
    filepath = folder / filename
    target = "/mmmjjjjjjjjjj/"              
    targetfile = target + filename  
    d = dropbox.Dropbox('kNql765KtvQAAAAAAAAAAUiTzTr2uKcGs5yC58H1J1XhuF2dujH7f8RCqjzFH3N9')
    with filepath.open("rb") as f:
        meta = d.files_upload(f.read(), targetfile, mode=dropbox.files.WriteMode("overwrite"))
    session.quit()

def image_pro():
    image = cv2.imread('given_user.jpg')
    img = cv2.resize(image, (1200,800), interpolation = cv2.INTER_AREA)
    def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
        b = (255 - brightness)/255
        buf = cv2.addWeighted(input_img, b, input_img, 0, brightness)
        f = 131*(contrast + 127)/(127*(131-contrast))
        c = 127*(1-f)
        buf = cv2.addWeighted(buf,f, buf, 0,c)
        return buf
    
    out = np.zeros((1200, 1000), dtype = np.uint8)
    out = apply_brightness_contrast(img, 40, 50)
    cv2.imwrite('out.png', out)
    (thresh, gray) = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    img = gray[:-20,:-20] 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)) # Perform noise filtering
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    crop = img[y:y+h, x:x+w] 
    cv2.imwrite("rect.png", crop) 
    return crop

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def mainline():
    image=image_pro()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite("rect2.png", thresh) 
    kernel = np.ones((7,60), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sort_contours(ctrs, method="left-to-right")[0]
    ctrs = sorted(ctrs,key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img_dilation.shape[1])

    rect_areas = []
    for ctr in (ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        rect_areas.append(w * h)
    avg_area = mean(rect_areas)
    print(avg_area)
    
    if os.path.exists("LineOutput"):
        files = glob.glob("LineOutput/*")
        for f in files:
            os.remove(f)
    else:
        os.mkdir("LineOutput")
    k=0
    for i,ctr in enumerate (ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        cnt_area = w * h
        if cnt_area > 0.5 * avg_area:
            roi = image[y:y+h, x:x+w]
            cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,255),2)
            img = Image.fromarray(roi)
            img = img.resize((1000, 600))
            img.save("LineOutput/img{0}.png".format(k))
            k=k+1
    mainpdf()

@app.route('/predict/',methods=["GET","POST"])
def predict():
    if request.method=="POST":
        qtc_data = request.get_json()
        imgstring=(str(qtc_data))
        imgstring=(imgstring.replace('<img src="data:image/jpeg;base64,',''))
        imgstring += "=" * ((4 - len(imgstring) % 4) % 4) 
        imgdata = base64.b64decode(imgstring)
        filename = 'given_user.jpg'  
        with open(filename, 'wb') as f:
            f.write(imgdata)
        mainline()
    return render_template('index.html')


if __name__=='__main__':
	app.run(host="0.0.0.0")