import cv2
import imageio
import glob
import numpy as np


ear_cascade = cv2.CascadeClassifier('cascades/cascade-lbp-s23-p550-n3000-w24-h48.xml')
scale_step = 1.01
size = 2


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


def detect_ears(skip_rgb=True):
    accuracies = []
    precisions = []
    recalls = []
    IoUs = []
    for im_path in glob.glob("../AWEForSegmentation/test/*.png"):
        im_name = im_path.split("../AWEForSegmentation/test\\")[1]
        img = cv2.imread(im_path)
        # Uncomment to show the image with the drawn boxes
        # imgToDraw = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ears = ear_cascade.detectMultiScale(gray, scale_step, size)
        boxes = []
        for (x, y, w, h) in ears:
            red = np.mean(img[y:y+h, x:x+w, 2])
            green = np.mean(img[y:y+h, x:x+w, 1])
            blue = np.mean(img[y:y+h, x:x+w, 0])
            if skip_rgb or red > green and red > blue:
                boxes.append(DetectedBoxes(x, y, w, h))
                # Uncomment to show the image with the drawn boxes
                # cv2.rectangle(imgToDraw, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Uncomment to show the image with the drawn boxes
        #cv2.imshow('img', imgToDraw)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
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
    f_accuracies, f_precisions, f_recalls, f_IoUs = detect_ears(True)
    a = np.asarray(f_accuracies)
    p = np.asarray(f_precisions)
    r = np.asarray(f_recalls)
    i = np.asarray(f_IoUs)
    print("Accuracy mean: " + str(a.mean()) + " st.dev.: " + str(a.std()))
    print("Precision mean: " + str(p.mean()) + " st.dev.: " + str(p.std()))
    print("Recall mean: " + str(r.mean()) + " st.dev.: " + str(r.std()))
    print("IoU mean: " + str(i.mean()) + " st.dev.: " + str(i.std()))
