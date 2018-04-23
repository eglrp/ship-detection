import caffe

class PatchTransLayer(caffe.Layer):

    def setup(self, bottom, top):

        top[0].reshape(42*31, 9)

    def forward(self, bottom, top):
        # here we reshape the data of bottom layer with size (1, 9, 31, 42) to (42*31, 9)
        # it is the same as the reshape layer during training.
        #(1, 9, 31, 42)
        all_feature_map = bottom[0].data
        #(1, 31, 42, 9)
        channel_swap = (0, 2, 3, 1)
        all_feature_map = all_feature_map.transpose(channel_swap)
        top[0].data[:,:] = all_feature_map.reshape(42*31, 9)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass