import os
import tensorflow as tf
import numpy as np
import cv2

import pandas as pd
from pytrends.request import TrendReq

from fbprophet import Prophet
import matplotlib.pyplot as plt

def predictImage(openCVImage):
    # module-level variables ##############################################################################################
    RETRAINED_LABELS_TXT_FILE_LOC = os.path.dirname(os.path.abspath(__file__)) + "/" + "retrained_labels.txt"
    RETRAINED_GRAPH_PB_FILE_LOC = os.path.dirname(os.path.abspath(__file__)) + "/" + "retrained_graph.pb"

    # TEST_IMAGES_DIR = os.getcwd() + "/test_images"

    SCALAR_RED = (0.0, 0.0, 255.0)
    SCALAR_BLUE = (255.0, 0.0, 0.0)

    print("starting prediction program . . .")

    if not os.path.exists(RETRAINED_LABELS_TXT_FILE_LOC):
        print('ERROR: RETRAINED_LABELS_TXT_FILE_LOC "' + RETRAINED_LABELS_TXT_FILE_LOC + '" does not seem to exist')
        return
    if not os.path.exists(RETRAINED_GRAPH_PB_FILE_LOC):
        print('ERROR: RETRAINED_GRAPH_PB_FILE_LOC "' + RETRAINED_GRAPH_PB_FILE_LOC + '" does not seem to exist')
        return

    # get a list of classifications from the labels file
    classifications = []
    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        classification = currentLine.rstrip()
        classifications.append(classification)
    print("classifications = " + str(classifications))

    # load the graph from file
    with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        graphDef = tf.GraphDef()
        graphDef.ParseFromString(retrainedGraphFile.read())
        _ = tf.import_graph_def(graphDef, name='')

    with tf.Session() as sess:

        # get the final tensor from the graph
        finalTensor = sess.graph.get_tensor_by_name('final_result:0')
        tfImage = np.array(openCVImage)[:, :, 0:3]
        predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

        # sort predictions from most confidence to least confidence
        sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]
        print("---------------------------------------")
        # for each prediction . . .
        for prediction in sortedPredictions:
            strClassification = classifications[prediction]

            # if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
            if strClassification.endswith("s"):
                strClassification = strClassification[:-1]
            # get confidence, then get confidence rounded to 2 places after the decimal
            confidence = predictions[0][prediction]
            # get the score as a %
            scoreAsAPercent = confidence * 100.0
            # show the result to std out
            print("the object appears to be a " + strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
            return strClassification, scoreAsAPercent
                
    return None, None

# pip install pytrends
# pip install Cython
# pip install fbprophet

def searchTrends(words):
    print("Starting Google API...")
    pt = TrendReq(hl='en-US', tz = 360)
    print("Loading key words...")
    pt.build_payload(words, cat=0, timeframe='today 3-m', geo='MX', gprop='')
    print("Getting data...")
    data = pt.interest_over_time()
    return data


def formatTrendData(data):
    data.drop(columns=['isPartial'], inplace=True)
    data.reset_index(inplace = True)
    data.columns = ["ds","y"]
    return data


def prophet(data):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=30)
    future.tail()
    forecast = model.predict(future)

    fig = model.plot_components(forecast)
    return forecast, fig

def runPrediction(image):
    category, score = predictImage(image) # Actually predicts all image from the first image in the testing directory
    print("---------------------------------------------")
    print(category)
    print(score)
    print("---------------------------------------------")
    data = searchTrends([category])
    fdata = formatTrendData(data)
    prediction, fig = prophet(fdata)

    

    return category, score, prediction, fig
