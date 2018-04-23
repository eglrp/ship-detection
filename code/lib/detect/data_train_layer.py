import caffe
from detect.config import cfg
import numpy as np
import cv2
import random

def get_images_blob(selected_ind, roidb):

    num_images = len(selected_ind)
    ind = 0

    data_blob = np.zeros( (cfg.BATCH_SIZE, cfg.INPUT_IMAGE_HEIGHT,cfg.INPUT_IMAGE_WIDTH,3), dtype=np.float32)
    roi_blob = np.zeros((num_images, 5), dtype=np.float32)
    label = np.zeros((num_images, 1), dtype=np.int16)
    bbox_target_blob = np.zeros((num_images, 4), dtype=np.float32)
    bbox_inside_weights = np.zeros((num_images, 4), dtype=np.float32)
    bbox_outside_weights = np.zeros((num_images, 4), dtype=np.float32)

    for i in selected_ind:
        im = cv2.imread(roidb[i]['image'])
        im = im.astype(np.float32, copy=False)
        im -= cfg.IMAGE_MEANS
        im *= cfg.IMAGE_SCALE 
        roi = roidb[i]['roi']
        bbox_target = roidb[i]['bbox_target']
        label[ind,0] = roidb[i]['label'] 

        data_blob[ind, 0:im.shape[0], 0:im.shape[1], :] = im
        roi_blob[ind, 1:roi.shape[0]+1] = roi
        roi_blob[ind, 0] = ind
        bbox_target_blob[ind, 0:bbox_target.shape[0]] = bbox_target
        bbox_inside_weights[ind, :] = cfg.BBOX_INSIDE_WEIGHTS
        ind += 1

    channel_swap = (0, 3, 1, 2)
    data_blob = data_blob.transpose(channel_swap)

    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
        
    blobs = {'data': data_blob}
    blobs['roi'] = roi_blob
    blobs['label'] = label
    blobs['bbox_target'] = bbox_target_blob
    blobs['bbox_inside_weights'] = bbox_inside_weights   
    blobs['bbox_outside_weights'] = bbox_outside_weights  

    return blobs

def get_oneimage_blob(selected_ind, roidb):

    num_images = len(selected_ind)
    ind = 0

    data_blob = np.zeros((1, cfg.INPUT_IMAGE_HEIGHT,cfg.INPUT_IMAGE_WIDTH,3), dtype=np.float32)
    roi_blob = np.zeros((num_images, 5), dtype=np.float32)
    label = np.zeros((num_images, 1), dtype=np.int16)
    bbox_target_blob = np.zeros((num_images, 4), dtype=np.float32)
    bbox_inside_weights = np.zeros((num_images, 4), dtype=np.float32)
    bbox_outside_weights = np.zeros((num_images, 4), dtype=np.float32)
        
    for i in selected_ind:
        roi = roidb[i]['roi']
        bbox_target = roidb[i]['bbox_target']
        label[ind,0] = roidb[i]['label']

        roi_blob[ind, 1:roi.shape[0]+1] = roi
        roi_blob[ind, 0] = 0
        bbox_target_blob[ind, 0:bbox_target.shape[0]] = bbox_target
        bbox_inside_weights[ind, :]  = cfg.BBOX_INSIDE_WEIGHTS
        bbox_outside_weights[ind, :] = cfg.BBOX_INSIDE_WEIGHTS

        ind += 1

    im = cv2.imread(roidb[selected_ind[0]]['image'])
    im = im.astype(np.float32, copy=False)
    im -= cfg.IMAGE_MEANS
    im *= cfg.IMAGE_SCALE 
    data_blob[0, 0:im.shape[0], 0:im.shape[1],:] = im
    channel_swap = (0, 3, 1, 2)
    data_blob = data_blob.transpose(channel_swap)

    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    blobs = {'data': data_blob}
    blobs['roi'] = roi_blob
    blobs['label'] = label
    blobs['bbox_target'] = bbox_target_blob
    blobs['bbox_inside_weights'] = bbox_inside_weights   
    blobs['bbox_outside_weights'] = bbox_outside_weights  
       
    return blobs

class RoIDataLayer(caffe.Layer):
    """data layer used for training."""
    def set_roidb(self, roidb, index_record):
        self._index_record = index_record
        self._roidb = roidb
#        self._perm = np.random.permutation(np.arange(len(self._roidb)))        

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""
#        self._cur = 0
#        self._perm = {}
        self._roidb = {}
        self._index_record = {}
#        layer_params = yaml.load(self.param_str_)
#        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 1 channels
#        top[0].reshape(cfg.BATCH_SIZE, 3, cfg.INPUT_IMAGE_HEIGHT, cfg.INPUT_IMAGE_WIDTH  )
        top[0].reshape(1, 3, cfg.INPUT_IMAGE_HEIGHT, cfg.INPUT_IMAGE_WIDTH )
        self._name_to_top_map['data'] = 0

        top[1].reshape(cfg.BATCH_SIZE, 5)
        self._name_to_top_map['roi'] = 1

        top[2].reshape(cfg.BATCH_SIZE, 1)
        self._name_to_top_map['label'] = 2

        top[3].reshape(cfg.BATCH_SIZE, 4)
        self._name_to_top_map['bbox_target'] = 3

        # bbox_inside_weights
        top[4].reshape(cfg.BATCH_SIZE, 4)
        self._name_to_top_map['bbox_inside_weights'] = 4
        # bbox_outside_weights
        top[5].reshape(cfg.BATCH_SIZE, 4)
        self._name_to_top_map['bbox_outside_weights'] = 5

    def forward(self, bottom, top):

        while True:
#            imageindex = random.randint(1, cfg.IMAGE_TOATL );
            imageindex = np.array(random.choice(cfg.TABLE), dtype = int)
            # 12 = 2400/200  
            imageindex = imageindex + (np.array(random.randint(1, 12), dtype = int)-1) * (cfg.IMAGE_TOATL/12)

            begin = self._index_record.index(imageindex, 0)
            finish = self._index_record.index(imageindex+1, begin)
        
            methodList_ind = np.arange(begin, finish, 1)

            if len(methodList_ind) > 0:
                break

        if len(methodList_ind) > cfg.BATCH_SIZE:
            selected_ind = np.random.choice(methodList_ind, cfg.BATCH_SIZE)
        else:
            selected_ind = methodList_ind

        blobs = get_oneimage_blob(selected_ind, self._roidb)

        for blob_name, blob in blobs.iteritems():
              
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)
        '''

        if self._cur + cfg.BATCH_SIZE >= len(self._roidb):
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
            self._cur = 0

        selected_ind = self._perm[self._cur:self._cur + cfg.BATCH_SIZE]
        self._cur += cfg.BATCH_SIZE

        blobs = get_images_blob(selected_ind, self._roidb)

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)
        '''

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
