import glob
import cv2
import imageio
import numpy as np
import scipy.ndimage.measurements as mnts


# Structure to easily detect bounding boxes with mnts.find_objects
structure = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
])


# Prepares lines for annotations of positive image to be saved in ear_positives.info
def prepare_positive_info():
    info_lines = []
    for im_path in glob.glob("../AWEForSegmentation/trainannot_rect/*.png"):
        im = imageio.imread(im_path)
        bbox_slices = mnts.find_objects(mnts.label(im, structure=structure)[0])
        bboxes = []
        for bbox in bbox_slices:
            y_min = bbox[0].start
            # dejanski konec je pri bbox[0].stop - 1 ampak rabimo za racunat sirino/visino
            y_max = bbox[0].stop - 1
            x_min = bbox[1].start
            # dejanski konec je pri bbox[1].stop - 1 ampak rabimo za racunat sirino/visino
            x_max = bbox[1].stop - 1
            # x_min, x_max, y_min, y_max, width, height
            bboxes.append((x_min, x_max, y_min, y_max, x_max - x_min, y_max-y_min))
        im_name = im_path.split("../AWEForSegmentation/trainannot_rect\\")[1]
        line = "train/"+im_name + " " + str(len(bboxes))
        for bbox in bboxes:
            line += " " + str(bbox[0]) + " " + str(bbox[2]) + " " + str(bbox[4]) + " " + str(bbox[5])
        info_lines.append(line)
    return info_lines


# Prepares lines for annotations of negative images to be saved in ear_negatives.info
def prepare_negative_info():
    info_lines = []
    for im_path in glob.glob("../AWEForSegmentation/negatives/*"):
        im_name = im_path.split("negatives\\")[1]
        info_lines.append("negatives/"+im_name)
    return info_lines


# Converts train images into grayscale - used for training cascades
def convert_images():
    for im_path in glob.glob("../AWEForSegmentation/train/*.png"):
        img = cv2.imread(im_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_file = im_path.replace("train", "train_gray")
        cv2.imwrite(img_gray_file, gray)


if __name__ == '__main__':
    print("Preparing positive infos")
    positive_lines = prepare_positive_info()
    with open('ear_positives.info', 'w') as f:
        for item in positive_lines:
            f.write("%s\n" % item)
    print("Preparing positive infos DONE")
    print("Preparing negative infos...")
    negative_lines = prepare_negative_info()
    with open('ear_negatives.info', 'w') as f:
        for item in negative_lines:
            f.write("%s\n" % item)
    print("Preparing negative infos DONE")
    print("Converting train images into grayscale...")
    convert_images()
    print("Converting train images into grayscale DONE")
