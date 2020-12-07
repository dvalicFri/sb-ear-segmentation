import os
# set os.environ['CUDA_VISIBLE_DEVICES'] = '0' if you wish to execute on a CUDA GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import glob
import cv2
import imageio
from numpy import expand_dims
from mrcnn.model import mold_image
import numpy as np
from mrcnn.model import MaskRCNN
from train_keras import EarConfig


model_path = 'mask_rcnn_ear_cfg_0017.h5'


class DetectedBoxes:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def calc_tp_tn_fp_fn(rectangles, reference_image):
    im = imageio.imread("../AWEForSegmentation/testannot_rect/"+reference_image)
    detected = np.empty(im.shape)
    for rect in rectangles:
        for i in range(rect.x, rect.x + rect.w):
            for j in range(rect.y, rect.y + rect.h):
                detected[j][i] = 255
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for x in range(detected.shape[0]):
        for y in range(detected.shape[1]):
            if detected[x][y] == 255 and im[x][y] == 255:
                tp += 1
            elif detected[x][y] == 0 and im[x][y] == 0:
                tn += 1
            elif detected[x][y] == 0 and im[x][y] == 255:
                fn += 1
            elif detected[x][y] == 255 and im[x][y] == 0:
                fp += 1
    return tp, tn, fp, fn


def detect_ears():
    cfg = EarConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    # load model weights
    model.load_weights(model_path, by_name=True)
    accuracies = []
    precisions = []
    recalls = []
    IoUs = []
    for im_path in glob.glob("../AWEForSegmentation/test/*.png"):
        img = cv2.imread(im_path)
        # imgToDraw = img
        scaled_image = mold_image(img, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        ears = model.detect(sample, verbose=0)[0]
        boxes = []
        for box in ears['rois']:
            y, x, y_max, x_max = box
            w, h = x_max - x, y_max - y
            boxes.append(DetectedBoxes(x, y, w, h))
            # Uncomment to show the image with the drawn boxes
            # cv2.rectangle(imgToDraw, (x, y), (x_max, y_max), (255, 0, 0), 2)
        # Uncomment to show the image with the drawn boxes
        # cv2.imshow('img', imgToDraw)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        im_name = im_path.split("../AWEForSegmentation/test\\")[1]
        tp, tn, fp, fn = calc_tp_tn_fp_fn(boxes, im_name)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracies.append(accuracy)
        if tp + fp is not 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        precisions.append(precision)
        if tp + fn is not 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        recalls.append(recall)
        Iou = tp / (tp + fn + fp)
        IoUs.append(Iou)
        print("Image: " + im_name + " Accuracy: " + str(accuracy) + " Precision: " + str(precision) + " Recall: " + str(recall) + " IoU: " + str(Iou))
    return accuracies, precisions, recalls, IoUs


if __name__ == '__main__':
    f_accuracies, f_precisions, f_recalls, f_IoUs = detect_ears()
    a = np.asarray(f_accuracies)
    p = np.asarray(f_precisions)
    r = np.asarray(f_recalls)
    i = np.asarray(f_IoUs)
    print("Accuracy mean:" + str(a.mean()) + " st.dev.: "+str(a.std()))
    print("Precision mean:" + str(p.mean()) + " st.dev.: "+str(p.std()))
    print("Recall mean:" + str(r.mean()) + " st.dev.: "+str(r.std()))
    print("IoU mean:" + str(i.mean()) + " st.dev.: "+str(i.std()))
