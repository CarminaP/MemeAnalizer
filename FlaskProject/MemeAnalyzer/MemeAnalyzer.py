import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import io

from uuid import uuid4
from flask import Flask, request, render_template, send_from_directory

from prediction import runPrediction

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():

    #### SAVE IMAGE TO images/
    target = os.path.join(APP_ROOT, 'images/')
    plotTarget = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    files = request.files.getlist("file")
    print(files)
    upload = files[0]
    print(upload)
    print("{} is the file name".format(upload.filename))
    filename = upload.filename
    destination = "/".join([target, filename])
    print ("Accept incoming file:", filename)
    print ("Save it to:", destination)
    upload.save(destination)
    
    img = cv2.imread(destination)

    ##### PREDICTION
    category, pred = runPrediction(img)
    print(category)
    ##### EXAMPLE RGB ANALYSIS
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])

    plotName = "RGB.png"
    plt.savefig("/".join([plotTarget, plotName]))

    ######## LDA ANALYSIS (needs tensor flow output for <result>)
    
    result = category
    ldaResult = ""

    if result == "agujero negro":
        with io.open(APP_ROOT + '/LDA/agujero_negro.txt', 'r', encoding='latin-1') as myfile:
            ldaResult = myfile.read()
    elif result == "bob esponja":
        with io.open(APP_ROOT + '/LDA/bob_esponja.txt', 'r', encoding='latin-1') as myfile:
            ldaResult = myfile.read()
    elif result == "chavo del ocho":
        with io.open(APP_ROOT + '/LDA/chavo_del_ocho.txt', 'r', encoding='latin-1') as myfile:
            ldaResult = myfile.read()
    elif result == "dice mi mama":
        with io.open(APP_ROOT + '/LDA/dice_mi_mama.txt', 'r', encoding='latin-1') as myfile:
            ldaResult = myfile.read()
    elif result == "komo lo zupo":
        with io.open(APP_ROOT + '/LDA/komo_lo_zupo.txt', 'r', encoding='latin-1') as myfile:
            ldaResult = myfile.read()
    elif result == "los simpson":
        with io.open(APP_ROOT + '/LDA/los_simpson.txt', 'r', encoding='latin-1') as myfile:
            ldaResult = myfile.read()
    elif result == "pikachu sorprendido":
        with io.open(APP_ROOT + '/LDA/pikachu_sorprendido.txt', 'r', encoding='latin-1') as myfile:
            ldaResult = myfile.read()
    elif result == "se tenia que decir":
        with io.open(APP_ROOT + '/LDA/se_tenia_que_decir.txt', 'r', encoding='latin-1') as myfile:
            ldaResult = myfile.read()
    elif result == "ya nos exhibiste":
        with io.open(APP_ROOT + '/LDA/ya_nos_exhibistes.txt', 'r', encoding='latin-1') as myfile:
            ldaResult = myfile.read()
    return render_template("complete.html", image_name=filename, lda_result=ldaResult)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=8080, debug=True)