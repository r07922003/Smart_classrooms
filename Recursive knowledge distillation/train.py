from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import sklearn
import pickle
import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
from config import config, default, generate_config
from metric import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import flops_counter
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
import verification
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbol'))
import fresnet
import fmobilefacenet
import fmobilenet
import fmnasnet
import fdensenet
import vargfacenet
from mxnet.model import BatchEndParam
from mxnet.model import _multiple_callbacks

"""
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u change_recursive\ knowledge\ distillation\ _train.py --network vargfacenet  --loss arcface --dataset emore --pretrained models/lossvalue_15.529288/model --pretrained-epoch 1
"""
logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = None

def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--dataset', default=default.dataset, help='dataset config')
  parser.add_argument('--network', default=default.network, help='network config')
  parser.add_argument('--loss', default=default.loss, help='loss config')
  args, rest = parser.parse_known_args()
  generate_config(args.network, args.dataset, args.loss)
  parser.add_argument('--models-root', default=default.models_root, help='root directory to save model.')
  parser.add_argument('--pretrained', default=default.pretrained, help='pretrained model to load')
  parser.add_argument('--pretrained-epoch', type=int, default=default.pretrained_epoch, help='pretrained epoch to load')
  parser.add_argument('--ckpt', type=int, default=default.ckpt, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--verbose', type=int, default=default.verbose, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--lr', type=float, default=default.lr, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default=default.lr_steps, help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=default.wd, help='weight decay')
  parser.add_argument('--mom', type=float, default=default.mom, help='momentum')
  parser.add_argument('--frequent', type=int, default=default.frequent, help='')
  parser.add_argument('--per-batch-size', type=int, default=default.per_batch_size, help='batch size in each context')
  parser.add_argument('--kvstore', type=str, default=default.kvstore, help='kvstore setting')
  args = parser.parse_args()
  return args

def get_symbol(args):

    student_embedding = eval('vargfacenet').get_symbol()
    #Angular Distillation Loss
    """===============Teacher Model Setting==============="""
    teacher_embedding = mx.symbol.Variable('teacher_embedding')     #正確輸入
    student_nembedding = mx.symbol.L2Normalization(student_embedding, mode='instance', name='student_1n')  #正確
    teacher_nembedding = mx.symbol.L2Normalization(teacher_embedding, mode='instance', name='teacher_1n')  #正確
    KD_loss = teacher_nembedding - student_nembedding #(128,512)維
    KD_loss = mx.symbol.norm(KD_loss,ord=2,axis=1,name='KD_loss') #128維
    #KD_loss = mx.symbol.L2Normalization(KD_loss, mode='instance',name='KD_loss')
    """===============Teacher Model Setting==============="""

    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label
    is_softmax = True
    # Loss :Arcface Loss
    _weight = mx.symbol.Variable("fc7_weight", shape=(config.num_classes, config.emb_size),lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult, init=mx.init.Normal(0.01))
    #print("name =",_weight.name)
    #print("list_arguments =",_weight.list_arguments())
    s = config.loss_s   #64
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    nembedding = mx.symbol.L2Normalization(student_embedding, mode='instance', name='fc1n')*s
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=config.num_classes, name='fc7')
    if config.loss_m1!=1.0 or config.loss_m2!=0.0 or config.loss_m3!=0.0:
        zy = mx.sym.pick(fc7, gt_label, axis=1)
        cos_t = zy/s    #實際的cos值
        t = mx.sym.arccos(cos_t)
        if config.loss_m1!=1.0:
            t = t*config.loss_m1
        if config.loss_m2>0.0:
            t = t+config.loss_m2
        body = mx.sym.cos(t)
        if config.loss_m3>0.0:
            body = body - config.loss_m3
        new_zy = body*s
        diff = new_zy - zy
        diff = mx.sym.expand_dims(diff, 1)
        gt_one_hot = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = 1.0, off_value = 0.0)
        body = mx.sym.broadcast_mul(gt_one_hot, diff)
        fc7 = fc7+body
    out_list = [mx.symbol.BlockGrad(student_embedding)]
    if is_softmax:
        softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid')
        out_list.append(softmax)
        if config.ce_loss:
            body = mx.symbol.SoftmaxActivation(data=fc7)
            body = mx.symbol.log(body)
            _label = mx.sym.one_hot(gt_label, depth = config.num_classes, on_value = -1.0, off_value = 0.0)
            body = body*_label
            ce_loss = mx.symbol.sum(body)/args.per_batch_size + 7.0*mx.symbol.sum(KD_loss)/args.per_batch_size
            out_list.append(mx.symbol.BlockGrad(ce_loss))
    out = mx.symbol.Group(out_list)
    return out
  

def nealson_train_net(args):
  "=================Environment setting================"
  ctx = []
  cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
  if len(cvd)>0:
    for i in range(len(cvd.split(','))):
      ctx.append(mx.gpu(i))
  if len(ctx)==0:
    ctx = [mx.cpu()]
    print('use cpu')
  else:
    print('gpu num:', len(ctx))
  prefix = os.path.join(args.models_root, '%s-%s-%s'%(args.network, args.loss, args.dataset), 'model')
  prefix_dir = os.path.dirname(prefix)
  print('prefix', prefix)
  if not os.path.exists(prefix_dir):
    os.makedirs(prefix_dir)
  args.ctx_num = len(ctx)
  args.batch_size = args.per_batch_size*args.ctx_num  #128
  args.rescale_threshold = 0
  args.image_channel = config.image_shape[2] #3
  config.per_batch_size = args.per_batch_size

  #model 訓練、初始化參數設定
  initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
  _rescale = 1.0/args.ctx_num
  opt = optimizer.SGD(learning_rate=args.lr, momentum=args.mom, wd=args.wd, rescale_grad=_rescale)
  _cb = mx.callback.Speedometer(args.batch_size, args.frequent)

  metric1 = AccMetric()
  eval_metrics = [mx.metric.create(metric1)]
  if config.ce_loss:
      metric2 = LossValueMetric()
      eval_metrics.append( mx.metric.create(metric2) )

  "=================Data setting================"
  data_dir = config.dataset_path
  print(data_dir)
  path_imgrec = None
  path_imglist = None
  image_size = config.image_shape[0:2]
  assert len(image_size)==2
  assert image_size[0]==image_size[1]
  print('image_size', image_size)
  print('num_classes', config.num_classes)
  path_imgrec = os.path.join(data_dir, "train.rec")
  print(path_imgrec)

  print('Called with argument:', args, config)
  data_shape = (args.image_channel,image_size[0],image_size[1])
  mean = None

  "=====================input data====================="
  val_dataiter = None

  from image_iter import FaceImageIter
  train_dataiter = FaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrec          = path_imgrec,
        shuffle              = True,
        rand_mirror          = config.data_rand_mirror,
        mean                 = mean,
        cutoff               = config.data_cutoff,
        color_jittering      = config.data_color,
        images_filter        = config.data_images_filter,
  )
  train_dataiter = mx.io.PrefetchingIter(train_dataiter)

  "=========================================Load Teacher model=========================================="
  teacher_symbol, teacher_arg_params, teacher_aux_params = mx.model.load_checkpoint(config.teacher_prefix,config.teacher_epoch)
  #print(teacher_symbol.get_internals())
  teacher_symbol = teacher_symbol.get_internals()[config.teacher_symbol]
  teacher_module = mx.module.Module(teacher_symbol, context=ctx,label_names = None)
  teacher_module.bind(data_shapes = train_dataiter.provide_data, for_training=False, grad_req='null')
  teacher_module.set_params(teacher_arg_params, teacher_aux_params)
  print("======load teacher sucessful======")

  "========================================Load student model======================================="
  begin_epoch = 0
  if len(args.pretrained)==0:
    arg_params = None
    aux_params = None
    sym = get_symbol(args)
    #print(sym.list_arguments())
  else:
    print('loading', args.pretrained, args.pretrained_epoch)
    _, arg_params, aux_params = mx.model.load_checkpoint(args.pretrained, args.pretrained_epoch)
    sym = get_symbol(args)
  """===debug專用==="""
  #internals = sym.get_internals()
  #print(internals)
  #te1n = internals['teacher_1n_output']
  #st1n = internals['student_1n_output']
  #KD_loss = internals['KD_loss_output']
  #group = mx.symbol.Group([sym,te1n, st1n,KD_loss])
 
  student_model = mx.mod.Module(
    context       = ctx,
    symbol        = sym, #正常用sym debug用group
    data_names    = ['data','teacher_embedding'],
    label_names    = ['softmax_label'],
    )
  student_model.bind(
    data_shapes        = [('data', (args.batch_size ,3,112,112)),('teacher_embedding', (args.batch_size,512))],
    label_shapes       = [('softmax_label', (args.batch_size,))],
    for_training       = True
  )
  #print("arg_params:",len(arg_params))
  #print("aux_params",len(aux_params))
  student_model.init_params(
    initializer        = initializer,
    arg_params         = arg_params,
    aux_params         = aux_params,
    allow_missing      = True
  )
  
  student_model.init_optimizer(
    kvstore=args.kvstore,
    optimizer=opt
    #optimizer_params=optimizer_params
    )
  
  print("======load student sucessful======")
  "======================training set======================="
  global_step = [0]
  save_step = [0]
  lr_steps = [int(x) for x in args.lr_steps.split(',')]
  print('lr_steps', lr_steps)
  #print(eval_metrics)
  if not isinstance(eval_metrics, mx.metric.EvalMetric):
    eval_metric = mx.metric.create(eval_metrics)
  "======================define batch_callback======================"
  def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]
      for step in lr_steps:
        if mbatch==step:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break

      _cb(param)
      if mbatch%1000==0:
        print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)

      #if mbatch>=0 and mbatch%args.verbose==0:
      if mbatch>=0 and mbatch%20==0:
        #acc_list = ver_test(mbatch)
        save_step[0]+=1
        msave = save_step[0]
        do_save = True
        is_highest = False
        """
        if len(acc_list)>0:
          #lfw_score = acc_list[0]
          #if lfw_score>highest_acc[0]:
          #  highest_acc[0] = lfw_score
          #  if lfw_score>=0.998:
          #    do_save = True
          score = sum(acc_list)
          if acc_list[-1]>=highest_acc[-1]:
            if acc_list[-1]>highest_acc[-1]:
              is_highest = True
            else:
              if score>=highest_acc[0]:
                is_highest = True
                highest_acc[0] = score
            highest_acc[-1] = acc_list[-1]
            #if lfw_score>=0.99:
            #  do_save = True
        """
        if is_highest:
          do_save = True
        if args.ckpt==0:
          do_save = False
        elif args.ckpt==2:
          do_save = True
        elif args.ckpt==3:
          msave = 1

        if do_save:
          print('saving', msave)
          arg, aux = student_model.get_params()
          
          
          if config.ckpt_embedding:
            all_layers = student_model.symbol.get_internals()
            _sym = all_layers['fc1_output']
            _arg = {}
            for k in arg:
              if not k.startswith('fc7'):
                _arg[k] = arg[k]
            mx.model.save_checkpoint(prefix, msave, _sym, _arg, aux)
          
          else:
            mx.model.save_checkpoint(prefix, msave, student_model.symbol, arg, aux)
        
        #print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))
      if config.max_steps>0 and mbatch>config.max_steps:
        sys.exit(0)

  "======================start training======================="
  for epoch in range(begin_epoch,999999):
    eval_metric.reset()
    nbatch = 0
    data_iter = iter(train_dataiter)
    next_data_batch = next(data_iter)
    end_of_batch = False
    while not end_of_batch:
      data_batch = next_data_batch
      #teacher model predict
      next_teacher_input_data = mx.io.DataBatch(data=data_batch.data)
      teacher_module.forward(data_batch=next_teacher_input_data, is_train=False)
      teacher_embedding = teacher_module.get_outputs()
      #student model predict
      data_batch.data = data_batch.data + teacher_embedding
      student_model.forward(data_batch, is_train=True)
      student_model.backward()
      student_model.update()
      
      try:
        next_data_batch = next(data_iter)
      except StopIteration:
        end_of_batch = True

      student_model.update_metric(eval_metric, data_batch.label)
      
      if _batch_callback is not None:
        batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                          eval_metric=eval_metric,
                                          locals=locals())
        _multiple_callbacks(_batch_callback, batch_end_params)
      nbatch += 1

    train_dataiter.reset()
    for name, val in eval_metric.get_name_value():
      logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
    arg_params, aux_params = student_model.get_params()
    student_model.set_params(arg_params, aux_params)

      

  """
  ******************Dubug*******************
  "======================teacher predict debug 成功將train_dataiter取出batch.data再轉成mx.io.DataBatch格式========================="
  data_iter = iter(train_dataiter)
  #[(128, 3, 112, 112)] 128類、(RGB)、112x112
  next_data_batch = next(data_iter)
  next_teacher_input_data = mx.io.DataBatch(data=next_data_batch.data) 
  teacher_module.forward(data_batch=next_teacher_input_data, is_train=False)
  teacher_embedding = teacher_module.get_outputs()
  #print(teacher_embedding)     #<NDArray 128x512 @gpu(0)>]
  #next_data_batch.label = next_data_batch.label + teacher_embedding #將teacher embedding加入到label中 <NDArray 128 @cpu(0)>,<NDArray 128x512 @gpu(0)>
  next_data_batch.data = next_data_batch.data + teacher_embedding #將teacher embedding加入到data中 <NDArray 128x3x112x112 @cpu(0)>,<NDArray 128x512 @gpu(0)>
  student_model.forward(next_data_batch, is_train=True)
  student_model.backward()
  student_model.update()
  student_label = student_model.get_outputs()
  #print(student_label)         #[0]<NDArray 128x512 @gpu(0)> , [1]<NDArray 128x85742 @gpu(0)> , Loss [2]<NDArray 1 @gpu(0)>]
  #print(next_data_batch.label)  #<NDArray 128 @cpu(0)> 128維度的人臉編號
  ******************Dubug*******************
  """


def main():
  global args
  args = parse_args()
  nealson_train_net(args)
    

if __name__ == '__main__':
    main()
