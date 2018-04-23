import caffe
import numpy as np
from detect.config import cfg


class UpdateRoiLayer(caffe.Layer):

    def setup(self, bottom, top):
        # pooling rois (ind, x1, y1, x2, y2)
        top[0].reshape(cfg.PROPOSAL_NUMBER, 5)

    def forward(self, bottom, top):
        # previous rois (ind x1, y1, x2, y2)
        pre_rois = bottom[0].data
        # bbox_target
        gt_boxes = bottom[1].data

        now_blob5 = np.zeros( (len(pre_rois), 5), dtype=np.float32)

        now_blob5 = bbox_transform(pre_rois, gt_boxes, now_blob5)
        # new rois for pooling 
        top[0].reshape(*now_blob5.shape)
        top[0].data[...] = now_blob5

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
 
def bbox_transform(pre_roi, bbox, now_roi5):
#         shape:       1X5    1X4      1X5
    num_images = pre_roi.shape[0]

    for i in xrange(num_images):
        bbox[i][0] = bbox[i][0] * cfg.CENTERX_STD + cfg.CENTERX_MEAN
        bbox[i][1] = bbox[i][1] * cfg.CENTERY_STD + cfg.CENTERY_MEAN
        bbox[i][2] = bbox[i][2] * cfg.WIDTH_STD   + cfg.WIDTH_MEAN
        bbox[i][3] = bbox[i][3] * cfg.HEIGHT_STD  + cfg.HEIGHT_MEAN

        x = (pre_roi[i][3]+pre_roi[i][1])*0.5
        y = (pre_roi[i][4]+pre_roi[i][2])*0.5
        w = pre_roi[i][3]-pre_roi[i][1]
        h = pre_roi[i][4]-pre_roi[i][2]

        a = bbox[i][0]*w + x
        b = bbox[i][1]*h + y
        c = np.exp(bbox[i][2])*w
        d = np.exp(bbox[i][3])*h

        now_roi5[i][1] = a - c*0.5
        now_roi5[i][2] = b - d*0.5
        now_roi5[i][3] = a + c*0.5 
        now_roi5[i][4] = b + d*0.5     
        now_roi5[i][0] = 0
        
    return now_roi5