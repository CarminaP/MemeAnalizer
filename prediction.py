import os
import tensorflow as tf
import numpy as np
import cv2

import pandas as pd
from pytrends.request import TrendReq

from fbprophet import Prophet
import matplotlib.pyplot as plt

def predictImage():
    # module-level variables ##############################################################################################
    RETRAINED_LABELS_TXT_FILE_LOC = os.getcwd() + "/" + "retrained_labels.txt"
    RETRAINED_GRAPH_PB_FILE_LOC = os.getcwd() + "/" + "retrained_graph.pb"

    TEST_IMAGES_DIR = os.getcwd() + "/test_images"

    SCALAR_RED = (0.0, 0.0, 255.0)
    SCALAR_BLUE = (255.0, 0.0, 0.0)

    print("starting prediction program . . .")

    # detect that all the paths are ok 
    if not os.path.exists(TEST_IMAGES_DIR):
        print('')
        print('ERROR: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" does not seem to exist')
        print('Did you set up the test images?')
        print('')
        return
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

    # if the test image directory listed above is not valid, show an error message and bail
    if not os.path.isdir(TEST_IMAGES_DIR):
        print("the test image directory does not seem to be a valid directory, check file / directory paths")
        return
    
    with tf.Session() as sess:
        # for each file in the test images directory . . .
        for fileName in os.listdir(TEST_IMAGES_DIR):
            # if the file does not end in .jpg or .jpeg (case-insensitive), continue with the next iteration of the for loop
            if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
                continue
            print(fileName)

            # get the file name and full path of the current image file
            imageFileWithPath = os.path.join(TEST_IMAGES_DIR, fileName)
            openCVImage = cv2.imread(imageFileWithPath)

            # if we were not able to successfully open the image, continue with the next iteration of the for loop
            if openCVImage is None:
                print("unable to open " + fileName + " as an OpenCV image")
                continue

            # get the final tensor from the graph
            finalTensor = sess.graph.get_tensor_by_name('final_result:0')
            tfImage = np.array(openCVImage)[:, :, 0:3]
            predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

            # sort predictions from most confidence to least confidence
            sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]
            print("---------------------------------------")
            onMostLikelyPrediction = True
            # for each prediction . . .
            for prediction in sortedPredictions:
                strClassification = classifications[prediction]

                # if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
                if strClassification.endswith("s"):
                    strClassification = strClassification[:-1]
                # get confidence, then get confidence rounded to 2 places after the decimal
                confidence = predictions[0][prediction]

                # if we're on the first (most likely) prediction, state what the object appears to be and show a % confidence to two decimal places
                if onMostLikelyPrediction:
                    # get the score as a %
                    scoreAsAPercent = confidence * 100.0
                    # show the result to std out
                    print("the object appears to be a " + strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    return(strClassification)
                    onMostLikelyPrediction = False
                # for any prediction, show the confidence as a ratio to five decimal places
                print(strClassification + " (" +  "{0:.5f}".format(confidence) + ")")

    # write the graph to file so we can view with TensorBoard
    tfFileWriter = tf.summary.FileWriter(os.getcwd())
    tfFileWriter.add_graph(sess.graph)
    tfFileWriter.close()

    return strClassification

# pip install pytrends
# pip install Cython
# pip install fbprophet

def searchTrends(words):
    print("Starting API...")
    pt = TrendReq(hl='en-US', tz = 360)
    print("Loading key words...")
    pt.build_payload(words, cat=0, timeframe='today 3-m', geo='MX', gprop='')
    print("Getting data...")
    data = pt.interest_over_time()
    print(data)
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
    return forecast


prediction = predictImage() # Actually predicts all image from the first image in the testing directory
data = searchTrends([prediction])
fdata = formatTrendData(data)
prediction = prophet(fdata)
print(prediction)
plt.plot(prediction["trend"])
plt.show()




