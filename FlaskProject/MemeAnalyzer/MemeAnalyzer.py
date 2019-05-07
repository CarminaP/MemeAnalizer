import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import io
from sklearn.cluster import KMeans
from uuid import uuid4
from flask import Flask, request, render_template, send_from_directory

from prediction import runPrediction

def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

def rgb2hsv(colorarray_rgb):
    H = np.zeros(2)
    S = np.zeros(2)
    V = np.zeros(2)
    
    Percent  = np.zeros(2)
    
    for colors in np.arange(0,2):
        R = colorarray_rgb[colors][0]/255
        G = colorarray_rgb[colors][1]/255
        B = colorarray_rgb[colors][2]/255
        
        Percent[colors] = 100*hist[colors]/hist.sum()
        
        Max = max(R, G, B)
        Min = min(R, G, B)

        if Max == Min:
            H[colors] = 0
        elif Max == R:
            H[colors] = 60*((G-B)/(Max-Min))
        elif Max == G:
            H[colors] = 60*(2 + (B-R)/(Max-Min))
        elif Max == B:
            H[colors] = 60*(4 + (R-G)/(Max-Min))

        if H[colors] < 0:
            H[colors] += 360

        if Max == 0:
            S[colors] = 0
        else:
            S[colors] = 100*(Max-Min)/Max

        V[colors] = Max*100
        
    return zip(Percent, H, S, V)

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
    category, score, pred = runPrediction(img)
    print(category)

    plt.plot(pred["trend"])
    plotName = "Trend.png"
    plt.savefig("/".join([plotTarget, plotName]))
    
    ##### COLOR PROCESSING
    imgCol = np.array(img, dtype = np.uint8)
    imgCol = cv2.cvtColor(imgCol, cv2.COLOR_BGR2RGB)

    imgCol = imgCol.reshape((imgCol.shape[0] * img.shape[1], 3)) #represent as row*column, channel number
    clt = KMeans(n_clusters = 3) #cluster number
    clt.fit(imgCol)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    plt.axis("off")
    plt.imshow(bar)
    plotName = "Colors.png"
    plt.savefig("/".join([plotTarget, plotName]))

    ######## LDA ANALYSIS (needs tensor flow output for <result>)
    
    result = category
    ldaResult = ""

    if result == "agujero negro":
        with io.open(APP_ROOT + '/LDA/agujero_negro.txt', 'r', encoding='latin-1') as myfile:
            ldaResult = myfile.read()
    elif result == "bob Esponja memes":
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
    elif result == "los simpsons memes":
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
    return render_template("complete.html", image_name=filename, lda_result=ldaResult, category=category, score=score)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=8080, debug=True)
