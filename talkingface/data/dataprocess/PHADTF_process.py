import cv2
import os, sys
import glob
import dlib
import numpy as np
import time
import pdb
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../Deep3DFaceReconstruction/shape_predictor_68_face_landmarks.dat')

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def detect_image(imagename, savepath=""):
    image = cv2.imread(imagename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        eyel = np.round(np.mean(shape[36:42,:],axis=0)).astype("int")
        eyer = np.round(np.mean(shape[42:48,:],axis=0)).astype("int")
        nose = shape[33]
        mouthl = shape[48]
        mouthr = shape[54]
        if savepath != "":
            message = '%d %d\n%d %d\n%d %d\n%d %d\n%d %d\n' % (eyel[0],eyel[1],
                eyer[0],eyer[1],nose[0],nose[1],
                mouthl[0],mouthl[1],mouthr[0],mouthr[1])
            with open(savepath, 'w') as s_file:
                s_file.write(message)
            return
def detect_dir(folder):
    for file in sorted(glob.glob(folder+"/*.jpg")+glob.glob(folder+"/*.png")):
        print(file)
        detect_image(imagename=file, savepath=file[:-4]+'.txt')

t1 = time.time()
mp4 = sys.argv[1]
videoname = mp4
cap = cv2.VideoCapture(videoname)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#print videoname, length
success, image = cap.read()
postfix = ".png"
if not os.path.exists(mp4[:-4]):
    os.makedirs(mp4[:-4])
count = 0
while count<400:
    cv2.imwrite("%s/frame%d%s"%(mp4[:-4],count,postfix),image)
    success, image = cap.read()
    count += 1
detect_dir(mp4[:-4])
t2 = time.time()
