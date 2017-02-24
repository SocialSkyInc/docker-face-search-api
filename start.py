#!/usr/bin/env python

import cv2
import numpy as np
import openface
import os
import pickle

from flask import abort, Flask, jsonify, request


PATH = '/home/face_search_api'
CLASSIFIER_PATH = os.path.join(PATH, 'brain', 'classifier.pkl')


PREDICTIONS_THRESHOLD = os.environ.get('PREDICTIONS_THRESHOLD', 75)


# make brain
if not os.path.isfile(CLASSIFIER_PATH):
    os.system('%s/learn.sh' % PATH)


# load brain
with open(CLASSIFIER_PATH, 'r') as f:
    (le, clf) = pickle.load(f)
    models_path = '/' + os.path.join('root', 'openface', 'models')
    align = openface.AlignDlib(os.path.join(
        models_path, 'dlib', 'shape_predictor_68_face_landmarks.dat'
    ))
    imgDim = 96
    net = openface.TorchNeuralNet(
        os.path.join(models_path, 'openface', 'nn4.small2.v1.t7'),
        imgDim=imgDim, cuda=False
    )


# load web server
app = Flask(__name__)


# from openface (demos/classifier.py)
def getRep(imgPath, multiple=False):
    # start = time.time()
    # bgrImg = cv2.imread(imgPath)
    buff = np.fromstring(imgPath.getvalue(), dtype=np.uint8)  # load from memory
    bgrImg = cv2.imdecode(buff, 1)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    # if args.verbose:
    #     print("  + Original size: {}".format(rgbImg.shape))
    # if args.verbose:
    #     print("Loading the image took {} seconds.".format(time.time() - start))

    # start = time.time()

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        raise Exception("Unable to find a face: {}".format(imgPath))
    # if args.verbose:
    #     print("Face detection took {} seconds.".format(time.time() - start))

    reps = []
    for bb in bbs:
        # start = time.time()
        alignedFace = align.align(
            imgDim,  # args.imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        # if args.verbose:
        #     print("Alignment took {} seconds.".format(time.time() - start))
        #     print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

        # start = time.time()
        rep = net.forward(alignedFace)
        # if args.verbose:
        #     print("Neural network forward pass took {} seconds.".format(
        #         time.time() - start))
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


@app.route('/find', methods=['POST'])
def find():
    """List of faces in image from left to right"""
    if not request.files or not 'img' in request.files:
        abort(400)

    try:
        reps = getRep(request.files['img'].stream, True)
    except Exception as e:
        return jsonify([])

    result = []
    for r in reps:
        rep = r[1].reshape(1, -1)
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)

        if predictions[maxI] >= (PREDICTIONS_THRESHOLD / 100):
            result.append(le.inverse_transform(maxI))

    return jsonify(result)


if __name__ == '__main__':
    app.run('0.0.0.0', 8000)

