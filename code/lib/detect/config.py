import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.GPU_ID = 0

__C.MAX_ITERS = 50000

__C.SNAPSHOT_EPOH = 10000

# Root directory of project
__C.ROOT_DIR = os.getcwd()
# father_path   = os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
# grader_father = os.path.abspath(os.path.dirname(pwd)+os.path.sep+"..")

__C.PRETRAINED_MODEL = None

__C.TRAIN_SOLVER = cfg.ROOT_DIR + '/solver.prototxt'

__C.SAVED_MODEL_NAME = cfg.ROOT_DIR + '/ship_s'

__C.BATCH_SIZE = 64

__C.TEST_MODEL = 'test.caffemodel'

__C.TEST_PROTOTXT = 'test.prototxt'

# Overlap threshold used for non-maximum suppression (suppress boxes with IoU >= this threshold)
__C.NMS = 0.5
__C.USE_GPU_NMS = False

# threshold to determine ship or non-ship 
__C.DETECT_THRESH = 0.95

__C.IMAGE_TYPE = '.jpg'

__C.IMAGE_DIR = cfg.ROOT_DIR + '/testimage'

__C.DATA_TXT = cfg.ROOT_DIR + '/trainimage/shipsamplestep2roi.txt'

__C.IMAGE_TOATL = 2400

__C.INPUT_IMAGE_WIDTH = 1024
__C.INPUT_IMAGE_HEIGHT = 768

__C.SHIPBODY_PATCH_WIDTH = 200
__C.SHIPBODY_PATCH_HEIGHT = 200

__C.PROPOSAL_NUMBER = 50

# Pixel mean values 127.5 = 255.0/2.0
__C.IMAGE_MEANS = np.array([[[127.5, 127.5, 127.5]]])

# 0.00784314 = 2.0/255.0
__C.IMAGE_SCALE = np.array([[[0.00784314, 0.00784314, 0.00784314]]])

__C.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# mean and standard deviation for x,y,w,h. Purpose is to make the regression targets having 0 mean and 1 variance
# note that: the mean and standard deviation would change if you restart generating new training samples.
# However, not changing them is also OK. Having 0 mean and 1 variance is not so strict requirement.
__C.CENTERX_STD  = 0.0683
__C.CENTERX_MEAN = -0.0014
__C.CENTERY_STD  = 0.0665
__C.CENTERY_MEAN = -8.0603e-04

__C.WIDTH_STD    = 0.3822
__C.WIDTH_MEAN   = -0.2902
__C.HEIGHT_STD   = 0.4365
__C.HEIGHT_MEAN  = -0.3215

__C.IMAGE_LIST = range(1, 200)
# the numbers correspond to numbers of ship im each training image. we assign higher weight to image with more ships
__C.IMAGE_WEIGHT = [4,4,4,2,7,6,8,8,8,8,8,6,8,12,9,6,7,11,6,5,6,7,12,8,7,6,10,8,10,5,6,5,12,4,6,8,7,6,6,6, \
                    6,7,6,6,6,9,5,6,6,6,10,6,10,10,8,6,13,6,6,8,8,8,5,7,6,7,6,6,6,7,7,5,5,5,7,7,6,6,7,7,9, \
                    10,8,11,10,10,8,10,9,10,8,7,11,9,6,8,9,6,6,8,7,6,8,9,9,8,8,8,11,9,4,7,6,7,7,4,17,13,12, \
                    13,21,9,11,7,12,11,13,12,13,7,10,10,10,10,8,7,7,8,8,7,5,9,9,8,9,7,8,11,8,12,13,12,9,11,8, \
                    8,8,10,8,9,11,13,7,7,8,9,13,8,8,8,5,6,7,10,8,6,9,13,9,13,8,8,13,13,8,7,13,8,5,5,5,4,4,4,5,5,10,7,6,8]
__C.TABLE = [z for x,y in zip(__C.IMAGE_LIST,__C.IMAGE_WEIGHT) for z in [x] * y]   