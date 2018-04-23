import numpy as np
import cv2
from detect.config import cfg
from detect.nms_wrapper import nms
import matplotlib.pyplot as plt
import os.path as osp
import bottleneck


def get_roi(label1):

    _roi_blob5 = np.zeros((42*31, 5), dtype=np.float32)

    len = 0
    # we apply sliding-window to classify ship heads. 42 and 31 stand for the numbers of window 
    # along horizontal and vertical directions with the input image size of 1024*768
    for i in xrange(31*42):

        label = label1[i]
        # xth in horizontal direction
        x = i%42
        # yth in vertical direction
        y = i/42

        if label == 1:
            _roi_blob5[len][1] = 1 + 24*x
            _roi_blob5[len][2] = 1 + 24*y - cfg.SHIPBODY_PATCH_HEIGHT/2 + 20
            len += 1
        elif label == 2: 
            _roi_blob5[len][1] = 1 + 24*x
            _roi_blob5[len][2] = 1 + 24*y 
            len += 1        
        elif label == 3:
            _roi_blob5[len][1] = 1 + 24*x - cfg.SHIPBODY_PATCH_WIDTH/2 + 20
            _roi_blob5[len][2] = 1 + 24*y
            len += 1
        elif label == 4: 
            _roi_blob5[len][1] = 1 + 24*x - cfg.SHIPBODY_PATCH_WIDTH + 40
            _roi_blob5[len][2] = 1 + 24*y
            len += 1    
        elif label == 5:
            _roi_blob5[len][1] = 1 + 24*x - cfg.SHIPBODY_PATCH_WIDTH + 40
            _roi_blob5[len][2] = 1 + 24*y - cfg.SHIPBODY_PATCH_HEIGHT/2 + 20
            len += 1
        elif label == 6:
            _roi_blob5[len][1] = 1 + 24*x - cfg.SHIPBODY_PATCH_WIDTH + 40
            _roi_blob5[len][2] = 1 + 24*y - cfg.SHIPBODY_PATCH_HEIGHT + 40
            len += 1               
        elif label == 7:
            _roi_blob5[len][1] = 1 + 24*x - cfg.SHIPBODY_PATCH_WIDTH/2 + 20
            _roi_blob5[len][2] = 1 + 24*y - cfg.SHIPBODY_PATCH_HEIGHT + 40
            len += 1
        elif label == 8:
            _roi_blob5[len][1] = 1 + 24*x
            _roi_blob5[len][2] = 1 + 24*y - cfg.SHIPBODY_PATCH_HEIGHT + 40
            len += 1

    new_roi_blob5 = np.zeros((len, 5), dtype=np.float32)

    for i in xrange(len):
        _roi_blob5[i][0] = 0
        _roi_blob5[i][3] = _roi_blob5[i][1] + cfg.SHIPBODY_PATCH_WIDTH
        _roi_blob5[i][4] = _roi_blob5[i][2] + cfg.SHIPBODY_PATCH_HEIGHT
        _roi_blob5[i][1] = max( 1, _roi_blob5[i][1] )
        _roi_blob5[i][2] = max( 1, _roi_blob5[i][2] )
        _roi_blob5[i][3] = min( cfg.INPUT_IMAGE_WIDTH, _roi_blob5[i][3] )
        _roi_blob5[i][4] = min( cfg.INPUT_IMAGE_HEIGHT, _roi_blob5[i][4] )

    new_roi_blob5[0:len,0:5] = _roi_blob5[0:len,0:5]

    if len == 0:
        print 'No ship, exit'
        exit()

    return new_roi_blob5

def im_detect(net, im, rois):

    pred_rois = list()
    prob_rois = list()

    blobs = {'data' : None , 'roi' : None}

    data_blob = np.zeros((1, cfg.INPUT_IMAGE_HEIGHT,cfg.INPUT_IMAGE_WIDTH,3), dtype=np.float32)
    data_blob[0, 0:im.shape[0], 0:im.shape[1],:] = im
    channel_swap = (0, 3, 1, 2)
    data_blob = data_blob.transpose(channel_swap)

    roi_blob = np.zeros((len(rois), 5), dtype=np.float32)
    roi_blob[:, :] = rois

    blobs['data'] = data_blob
    blobs['roi'] = roi_blob

#    print [(k, v) for k, v in net.blobs.items()]
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['roi'].reshape(*(blobs['roi'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['roi'] = blobs['roi'].astype(np.float32, copy=False)

    blobs_out = net.forward(**forward_kwargs)

    pred_rois = net.blobs['roi2'].data
    prob_rois = blobs_out['prob']

    return pred_rois, prob_rois

def im_detect_shiphead(net, im):

    blobs = {'data' : None}

    data_blob = np.zeros((1, cfg.INPUT_IMAGE_HEIGHT, cfg.INPUT_IMAGE_WIDTH, 3), dtype=np.float32)
    data_blob[0, 0:im.shape[0], 0:im.shape[1], :] = im
    channel_swap = (0, 3, 1, 2)
    data_blob = data_blob.transpose(channel_swap)

    blobs['data'] = data_blob

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}

    blobs_out = net.forward(**forward_kwargs)

    pred_rois = blobs_out['prob']

    label = np.zeros((31*42), dtype=np.int32)

    channel_swap = (1, 0)
    pred_rois_sort = pred_rois.transpose(channel_swap)[0]
    z = bottleneck.argpartition(pred_rois_sort, 0)[:cfg.PROPOSAL_NUMBER]
    for i in z:
        label[i] = np.argmax(pred_rois[i][1:9])+1

#    for i in xrange(31*42):
#        label[i] = np.argmax(pred_rois[i][1:9])+1
#        if pred_rois[i][0]>0.5:
#            label[i] = 0
    return label

def vis_detect_shiphead(im, label):
    """Visual debugging of detections."""
    plt.cla()    
#    plt.figure(figsize=(15,15)) 
    if im.ndim == 3:
        im = im[:,:,0]
    plt.imshow(im)

    for i in xrange(31*42):
        if label[i] > 0:
            x = i%42
            y = i/42
            plt.gca().add_patch(plt.Rectangle((1 + 24*x, 1 + 24*y), (40), (40), fill=False, edgecolor='g', linewidth=3))
    plt.show()
#    IMAGE_DIR = cfg.ROOT_DIR + '/result/resulthead'
#    plt.savefig(osp.join(IMAGE_DIR, str(cfg.IMAGE_NUMBER)+'.jpg'),bbox_inches='tight')

def vis_detections(im, bbox, prob, save):
    """Visual debugging of detections."""
    if save:
        IMAGE_DIR = cfg.ROOT_DIR + '/result/image'
        TXT_DIR   = cfg.ROOT_DIR + '/result/position'

    plt.cla()
#    plt.figure(figsize=(15,15))
    if im.ndim == 3:
        im = im[:,:,0]
    plt.imshow(im)
    for i in xrange(len(bbox)):
        plt.gca().add_patch(plt.Rectangle((bbox[i][1], bbox[i][2]), (bbox[i][3]-bbox[i][1]), (bbox[i][4]-bbox[i][2]), fill=False, edgecolor='g', linewidth=3) )                  
        plt.text(bbox[i][1], bbox[i][2] - 2,'{:.3f}'.format(float(prob[i,1])),fontdict={'size': 12, 'color': 'y'})
        if save:
            with open(osp.join(TXT_DIR, str(cfg.IMAGE_NUMBER)+'.txt'),"a") as f:
               new_con = str(np.int_(bbox[i][1])) + ' ' + str(np.int_(bbox[i][2])) + ' ' + str(np.int_(bbox[i][3])) + ' ' + str(np.int_(bbox[i][4])) + '\n'
               f.write(new_con)
    if save:
        plt.savefig(osp.join(IMAGE_DIR, str(cfg.IMAGE_NUMBER)+'.jpg'),bbox_inches='tight')
    else:
        plt.show()
        print bbox

def test_shiphead(net, image_source):

    im = cv2.imread(image_source)
    im = im.astype(np.float32, copy=False)

    im -= cfg.IMAGE_MEANS
    im *= cfg.IMAGE_SCALE 

    label = im_detect_shiphead(net, im)
    rois = get_roi( label)

    vis = False
    if vis:
        vis_detect_shiphead(cv2.imread(image_source).astype(np.float32, copy=False), label)

    return rois

def test_image(net, image_source, rois, vis=False, save=False):
    """Test network on a whole image."""

    im = cv2.imread(image_source)
#    im = cv2.imread(image_source, cv2.IMREAD_GRAYSCALE)
    im = im.astype(np.float32, copy=False)

    im -= cfg.IMAGE_MEANS
    im *= cfg.IMAGE_SCALE

    pred_rois, pred_prob = im_detect(net, im, rois)
    inds = np.where(pred_prob[:, 1] > cfg.DETECT_THRESH)[0]
    pred_rois = pred_rois[inds, :]
    pred_prob = pred_prob[inds, :]

    dets = np.zeros((len(pred_rois), 5), dtype=np.float32)

    dets[:,0:4] = pred_rois[:,1:5]
    dets[:,4] = pred_prob[:,1]

    keep = nms(dets, cfg.NMS)

#    vis = True
#    save= True
    if (vis or save):
        vis_detections(cv2.imread(image_source).astype(np.float32, copy=False), pred_rois[keep, :], pred_prob[keep, :], save)