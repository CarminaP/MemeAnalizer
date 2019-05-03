import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from uuid import uuid4
from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
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
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])

    plotName = "RGB.png"
    plt.savefig("/".join([plotTarget, plotName]))

    return render_template("complete.html", image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=8080, debug=True)