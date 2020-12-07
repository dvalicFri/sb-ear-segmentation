import imageio
import glob
import numpy as np
import scipy.ndimage.measurements as mnts

# Structure to easily detect bounding boxes with mnts.find_objects
structure = np.array([
    [1,1,1],
    [1,1,1],
    [1,1,1]
])


# Creates annotation files for keras, containing image size and one line for each bounding box
def create_annotations():
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
        data = str(im.shape[1]) + " " + str(im.shape[0])
        for bbox in bboxes:
            data += "\n" + str(bbox[0]) + " " + str(bbox[2]) + " " + str(bbox[1]) + " " + str(bbox[3])
        with open("annotations_keras/" + im_name.replace(".png", ".txt"), 'w') as f:
            f.write(data)


if __name__ == '__main__':
    create_annotations()
