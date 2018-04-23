import caffe
import numpy as np
import bottleneck
from detect.config import cfg


class ProposalLayer(caffe.Layer):

    def setup(self, bottom, top):
        # pooling rois (ind, x1, y1, x2, y2)
        top[0].reshape(cfg.PROPOSAL_NUMBER, 5)
        self.label = np.zeros((31*42), dtype=np.int32)
        self._roi_blob5 = np.zeros( (42*31, 5), dtype=np.float32)
  
    def forward(self, bottom, top):
        # shiphead classification probability  shape: 42*31, 9
        pred_rois = bottom[0].data
        channel_swap = (1, 0)
        pred_rois_sort = pred_rois.transpose(channel_swap)[0]
        z = bottleneck.argpartition(pred_rois_sort, 0)[:cfg.PROPOSAL_NUMBER]
        for i in z:
            self.label[i] = np.argmax(pred_rois[i][1:9])+1

        len = 0
        
        # we apply sliding-window to classify ship heads. 42 and 31 stand for the numbers of window 
        # along horizontal and vertical directions with the input image size of 1024*768
        for i in xrange(31*42):
            label = self.label[i]
            # xth in horizontal direction
            x = i%42
            # yth in vertical direction 
            y = i/42 

            if label == 1:
                self._roi_blob5[len][1] = 1 + 24*x
                self._roi_blob5[len][2] = 1 + 24*y - cfg.SHIPBODY_PATCH_HEIGHT/2
                len += 1
            elif label == 2: 
                self._roi_blob5[len][1] = 1 + 24*x
                self._roi_blob5[len][2] = 1 + 24*y 
                len += 1        
            elif label == 3:
                self._roi_blob5[len][1] = 1 + 24*x - cfg.SHIPBODY_PATCH_WIDTH/2
                self._roi_blob5[len][2] = 1 + 24*y
                len += 1
            elif label == 4:
                self._roi_blob5[len][1] = 1 + 24*x - cfg.SHIPBODY_PATCH_WIDTH
                self._roi_blob5[len][2] = 1 + 24*y
                len += 1    
            elif label == 5:
                self._roi_blob5[len][1] = 1 + 24*x - cfg.SHIPBODY_PATCH_WIDTH
                self._roi_blob5[len][2] = 1 + 24*y - cfg.SHIPBODY_PATCH_HEIGHT/2
                len += 1
            elif label == 6:
                self._roi_blob5[len][1] = 1 + 24*x - cfg.SHIPBODY_PATCH_WIDTH
                self._roi_blob5[len][2] = 1 + 24*y - cfg.SHIPBODY_PATCH_HEIGHT
                len += 1               
            elif label == 7:
                self._roi_blob5[len][1] = 1 + 24*x - cfg.SHIPBODY_PATCH_WIDTH/2
                self._roi_blob5[len][2] = 1 + 24*y - cfg.SHIPBODY_PATCH_HEIGHT
                len += 1
            elif label == 8:
                self._roi_blob5[len][1] = 1 + 24*x
                self._roi_blob5[len][2] = 1 + 24*y - cfg.SHIPBODY_PATCH_HEIGHT
                len += 1  

        for i in xrange(len):
            self._roi_blob5[i][0] = 0
            self._roi_blob5[i][3] = self._roi_blob5[i][1] + cfg.SHIPBODY_PATCH_WIDTH
            self._roi_blob5[i][4] = self._roi_blob5[i][2] + cfg.SHIPBODY_PATCH_HEIGHT
            self._roi_blob5[i][1] = max( 1, self._roi_blob5[i][1] )
            self._roi_blob5[i][2] = max( 1, self._roi_blob5[i][2] )
            self._roi_blob5[i][3] = min( cfg.INPUT_IMAGE_WIDTH, self._roi_blob5[i][3] )
            self._roi_blob5[i][4] = min( cfg.INPUT_IMAGE_HEIGHT, self._roi_blob5[i][4] )

        if len == 0:
            print 'No ship, exit()'
            exit()

        top[0].reshape(len, 5)

        top[0].data[0:len, 0:5] = self._roi_blob5[0:len, 0:5]
       
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
