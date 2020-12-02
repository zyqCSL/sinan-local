import os
import pdb

# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"
import mxnet as mx
import numpy as np
from mxnet.gluon import nn
import scipy.misc
counter = 0
#============== monitor =============#
class DMon(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x0 = in_data[0]
        x1 = in_data[1]
        o0 = np.round(x0.asnumpy())
        o1 = x1.asnumpy()
        '''
        o0 = x0.asnumpy().max(axis=1)
        o1 = x1.asnumpy().max(axis=1)
        qos = 100 * np.ones(shape=[o0.shape[0]])
        o0 = np.greater(o0, qos)
        o1 = np.greater(o1, qos)
        '''
        print np.sum(o0 == o1)*1.0/o0.shape[0]
        self.assign(out_data[0], req[0], x0)
        self.assign(out_data[1], req[0], x1)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], out_grad[0])

@mx.operator.register("DMon")
class DMonProp(mx.operator.CustomOpProp):
    def list_arguments(self):
        return ['data0', 'data1']
    def list_outputs(self):
        return ['output0', 'output1']
    def infer_shape(self, in_shapes):
        return [in_shapes[0], in_shapes[1]], [in_shapes[0], in_shapes[1]]
    def create_operator(self, ctx, in_shapes, in_dtypes):
        return DMon()
#============== monitor =============#
