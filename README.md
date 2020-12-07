# Ear detector

## Viola-Jones
The detector needs AWE original data-set to be present in [AWEForSegmentation](AWEForSegmentation/) directory. To run the detector usepython 3.7 and first install the required python packages listed in [requirements.txt](viola_jones/requirements.txt) and then execute [prepare_viola.py](viola_jones/prepare_viola.py).

## Keras
The detector needs AWE original data-set to be present in [AWEForSegmentation](AWEForSegmentation/) directory. Download [mask_rcnn_ear_cfg_0017.h5](https://github.com/Davidvster/sb-ear-segmentation/releases/download/1.0/mask_rcnn_ear_cfg_0017.h5) and put it in [keras_rcnn](keras_rcnn/) directory. To run the detector usepython 3.7 and first install the required python  packages liste in [requirements.txt](keras_rcnn/requirements.txt) and the https://github.com/akTwelve/Mask_RCNN library. After installing execute [detect_keras.py](keras_rcnn/detect_keras.py). To run with GPU, CUDA 10.1 and CuDNN for CUDA 10.1 are required. Also in [detect_keras.py](keras_rcnn/detect_keras.py) CUDA_VISIBLE_DEVICES must be changed to 0.

Note: For upload size reasons only model from 17th epoch was uploaded to github.

For the training download [mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) and put it in [keras_rcnn](keras_rcnn/) directory or start with the already downloaded [mask_rcnn_ear_cfg_0017.h5](https://github.com/Davidvster/sb-ear-segmentation/releases/download/1.0/mask_rcnn_ear_cfg_0017.h5).
