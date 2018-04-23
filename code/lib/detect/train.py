import caffe
from detect.config import cfg
from detect.timer import Timer
from caffe.proto import caffe_pb2
from google.protobuf import text_format

class SolverWrapper(object):

    def __init__(self, solver_prototxt, roidb, index_record, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb, index_record)

    def snapshot(self):

        net = self.solver.net

        filename = (cfg.SAVED_MODEL_NAME + '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        timer = Timer()

        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)
            if self.solver.iter % (cfg.SNAPSHOT_EPOH) == 0:
                self.snapshot()

def train_net(solver_prototxt, roidb, index_record, pretrained_model=None, max_iters=100000):
    print 'Train_net...'

    print solver_prototxt
    sw = SolverWrapper(solver_prototxt, roidb, index_record, pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)

    print 'done solving *************************************'