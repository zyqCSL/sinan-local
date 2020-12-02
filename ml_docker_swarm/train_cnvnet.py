# train cnn that predicts latency of next cycle
# param tunning
# python  train_cnvnet.py --num-examples 36304 --lr 0.001 --gpus 0,1 --data-dir ./swarm_data_next_5s --wd 0.001
# python  train_cnvnet.py --num-examples 58499 --lr 0.001 --gpus 0,1 --data-dir  ../logs/socialnet_ml_data/swarm_simple_sys_data_next_5s --wd 0.001

import mxnet as mx
import numpy as np
import math
import argparse,logging,os
from importlib import import_module

def multi_factor_scheduler(begin_epoch, epoch_size, step=[60, 75, 90], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    dst_dir = os.path.dirname(args.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d" % (
        args.model_prefix, rank))
    
def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def custom_metric(label, latency):
    return np.sqrt(np.mean(np.square(label - latency)))

def shuffle_in_unison(arr):
    rnd_state = np.random.get_state()
    for i in range(0, 10):
        for a in arr:
            np.random.set_state(rnd_state)
            np.random.shuffle(a)
            np.random.set_state(rnd_state)

# Can one get trained by brainwave? Thats needs investigated :)
def main():
    mx.random.seed(2333)
    np.random.seed(2333)
    data_dir = args.data_dir
 
    sys_data_t = np.load(data_dir + '/sys_data_train.npy')
    lat_data_t = np.load(data_dir + '/lat_data_train.npy')
    # # only use latency data of immediate future
    # nxt_data_t = np.squeeze(np.load(data_dir + '/nxt_k_data_train.npy')[:,:,:,0])
    # only cpu
    nxt_data_t = np.squeeze(np.load(data_dir + '/nxt_k_data_train.npy')[:,:,0])
    print(sys_data_t.shape, nxt_data_t.shape, lat_data_t.shape)

    label_t = np.squeeze(np.load(data_dir + '/nxt_k_train_label.npy')[:,:,0])
    print("label shape:", label_t.shape)
    qos = 500
    d = 505
    k = 0.01
    label_t = np.where(label_t < d, label_t, d+(label_t-d)/(1.0+k*(label_t-d)))

    train_data = {'data1':sys_data_t, 'data2':lat_data_t, 'data3':nxt_data_t} 
    train_label = {'label':label_t}
    
    sys_data_v = np.load(data_dir + '/sys_data_valid.npy')
    lat_data_v = np.load(data_dir + '/lat_data_valid.npy')
    # nxt_data_v = np.squeeze(np.load(data_dir + '/nxt_k_data_valid.npy')[:,:,:,0])
    nxt_data_v = np.squeeze(np.load(data_dir + '/nxt_k_data_valid.npy')[:,:,0])
    
    label_v = np.squeeze(np.load(data_dir + '/nxt_k_valid_label.npy')[:,:,0])
    label_v = np.where(label_v < d, label_v, d+(label_v-d)/(1.0+k*(label_v-d)))
    
    valid_data = {'data1':sys_data_v, 'data2':lat_data_v, 'data3':nxt_data_v} 
    valid_label = {'label':label_v}

    train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size=args.batch_size, shuffle=True)
    valid_iter = mx.io.NDArrayIter(valid_data, valid_label, batch_size=args.batch_size)
    
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol()

    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), 1)
    lr_scheduler = multi_factor_scheduler(0, epoch_size, step=[120, 150], factor=0.1)
    optimizer_params = {
            'learning_rate': args.lr,
            'wd': args.wd,
            'lr_scheduler': lr_scheduler}
    
    optimizer = 'nag'
    has_momentum = {'sgd', 'dcasgd', 'nag'}
    if optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom
    
    checkpoint = _save_model(args, kv.rank)

    eval_metric = mx.metric.CustomMetric(custom_metric, name='RMSE', output_names=['latency_output'],label_names=['label'])
    
    model   = mx.mod.Module(
            context = devs,
            symbol = sym,
            data_names = ('data1', 'data2', 'data3'),
            label_names = ('label',)
            )
    
    if args.load_epoch > 1:
        sym, arg_params, aux_params = _load_model(args, kv.rank)
        model.fit(
            train_iter,
            eval_data = valid_iter,
            optimizer_params = optimizer_params,
            optimizer = 'sgd',
            num_epoch = 1,
            arg_params = arg_params,
            epoch_end_callback=checkpoint,
            #initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2),
            # eval_metric = mx.metric.RMSE() 
            eval_metric = eval_metric
            )
    else:
        model.fit(
            train_iter,
            eval_data = valid_iter,
            optimizer_params = optimizer_params,
            optimizer = 'sgd',
            num_epoch = 200,
            epoch_end_callback=checkpoint,
            #initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2),
            # eval_metric = mx.metric.RMSE()
            eval_metric = eval_metric 
            )
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--lr', type=float, default=0.0005, help='initialization learning rate')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--bn-mom', type=float, default=0.9, help='momentum for batch normlization')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int, default=2048, help='the batch size')
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    parser.add_argument('--log', default='test_single_qps_upsample', type=str)
    parser.add_argument('--num-examples', type=int, required=True, help='the number of training examples')  # window size of 5
    parser.add_argument('--network', type=str, default='cnvnet')
    parser.add_argument('--model-prefix', type=str, default='./model/cnv')
    parser.add_argument('--load-epoch', type=int, default=0)
    parser.add_argument('--upsample', type=int, default=0)
    
    args = parser.parse_args()
    logging.basicConfig(filename=args.log+'.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.info(args)

    main()
