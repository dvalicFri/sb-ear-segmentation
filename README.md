# Ear detector

## Viola-Jones
The detector needs AWE original data-set to be present in AWEForSegmentation/. To run the detector usepython 3.7 and first install the required python packages listed in viola_jones/requirements.txt and then execute viola_jones/prepare_viola.py.

## Keras
The detector needs AWE original data-set to be present in AWEForSegmentation/. To run the detector usepython 3.7 and first install the required python  packages liste in keras_rcnn/requirements.txt and the https://github.com/akTwelve/Mask_RCNN library. After installing execute keras_rcnn/detect_keras.py. To run with GPU, CUDA 10.1 and CuDNN for CUDA 10.1 are required. Also in keras_rcnn/detect_keras.py CUDA_VISIBLE_DEVICES must be changed to 0.