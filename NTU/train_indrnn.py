import argparse
import numpy as np
import params
import indrnn
import cuda_indrnn

from __future__ import print_function
import sys
import os
import time
import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tools import stackedIndRNN_encoder
from eval_and_train import DataHandler_train, DataHandler_eval
from test import testDataHandler
from tools import U_bound
from matrix import get_confusion_matrix, plot_confusion_matrix, plt

parser = argparse.ArgumentParser(description='Train...')
parser.add_argument('cscv', help='cs or cv')
parser.add_argument('outputclass', type=int, help='the number of output class, can only be 2 or 60')

args = parser.parse_args()

if not (args.outputclass == 2 or args.outputclass == 60):
  print("error: output class can only be 2 or 60")
  exit(1)

if args.cscv == 'cv':
  test_CV = True
elif args.cscv == 'cs':
  test_CV = False
else:
  print("error: cscv can only be cs or cv")
  exit(2)

if test_CV:
  dropout = 0.1
else:
  dropout = 0.25
  

if test_CV:
  
  if args.outputclass is 60:
    train_datasets='./_60cls/cv_train_60cls'
    test_dataset='./_60cls/cv_test_60cls'
  else:
    train_datasets='./up_sampling_2cls/cv_train_2cls'
    test_dataset='./up_sampling_2cls/cv_test_2cls'

else:
  
  if args.outputclass is 60:
    train_datasets='./_60cls/cs_train_60cls'
    test_dataset='./_60cls/cs_test_60cls'
  else:
    train_datasets='./up_sampling_2cls/cs_train_2cls'
    test_dataset='./up_sampling_2cls/cs_test_2cls'



# Set the random seed manually for reproducibility.
seed=100
torch.manual_seed(seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed(seed)
  pass
else:
  print("WARNING: CUDA not available")

outputclass = params.outputclass
lr = params.lr
use_weightdecay_nohiddenW = params.use_weightdecay_nohiddenW
opti = params.opti
batch_size = params.batch_size
seq_len = params.seq_len
constrain_U = params.constrain_U
eval_fold = params.eval_fold
global_test_no = params.global_test_no
use_bneval = params.use_bneval
pThre = params.pThre
end_rate = params.end_rate
decayfactor = params.decay_factor
in_size=params.num_joints
gradientclip_value=10

model = stackedIndRNN_encoder(in_size, outputclass)  
model.cuda()
criterion = nn.CrossEntropyLoss()

#Adam with lr 2e-4 works fine.
learning_rate=lr
if use_weightdecay_nohiddenW:
  param_decay=[]
  param_nodecay=[]
  for name, param in model.named_parameters():
    if 'weight_hh' in name or 'bias' in name:
      param_nodecay.append(param)      
      #print('parameters no weight decay: ',name)          
    else:
      param_decay.append(param)      
      #print('parameters with weight decay: ',name)          

  if opti=='sgd':
    optimizer = torch.optim.SGD([
            {'params': param_nodecay},
            {'params': param_decay, 'weight_decay': decayfactor}
        ], lr=learning_rate,momentum=0.9,nesterov=True)   
  else:                
    optimizer = torch.optim.Adam([
            {'params': param_nodecay},
            {'params': param_decay, 'weight_decay': decayfactor}
        ], lr=learning_rate) 
else:  
  if opti=='sgd':   
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate,momentum=0.9,nesterov=True)
  else:                      
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
  
#from data_reader_numpy_witheval import DataHandler_train,DataHandler_eval  
#from data_reader_numpy_test import DataHandler as testDataHandler
dh_train = DataHandler_train(batch_size,seq_len)
dh_eval = DataHandler_eval(batch_size,seq_len)
dh_test= testDataHandler(batch_size,seq_len)
num_train_batches=int(np.ceil(dh_train.GetDatasetSize()/(batch_size+0.0)))
num_eval_batches=int(np.ceil(dh_eval.GetDatasetSize()/(batch_size+0.0)))
num_test_batches=int(np.ceil(dh_test.GetDatasetSize()/(batch_size+0.0)))
#print(num_train_batches)


def train(num_train_batches):
  model.train()
  tacc=0
  count=0
  start_time = time.time()
  for batchi in range(0,num_train_batches):
    inputs,targets=dh_train.GetBatch()
    inputs=inputs.transpose(1,0,2,3)
    #print(inputs.shape)
    
    inputs=Variable(torch.from_numpy(inputs).cuda(), requires_grad=True)
    targets=Variable(torch.from_numpy(np.int64(targets)).cuda(), requires_grad=False)

    model.zero_grad()
    if constrain_U:
      clip_weight(model,U_bound)
    output=model(inputs)
    loss = criterion(output, targets)

    pred = output.data.max(1)[1] # get the index of the max log-probability
    accuracy = pred.eq(targets.data).cpu().sum().numpy()/(0.0+targets.size(0))      
          
    loss.backward()
    clip_gradient(model,gradientclip_value)
    optimizer.step()
    
    tacc=tacc+accuracy#loss.data.cpu().numpy()#accuracy
    count+=1
  elapsed = time.time() - start_time
  print ("training accuracy: ", tacc/(count+0.0)  )
  #print ('time per batch: ', elapsed/num_train_batches)
  #print ('time per epoch: ', elapsed)
  
def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.train()       
def eval(dh,num_batches,use_bn_trainstat=False):
  model.eval()
  if use_bn_trainstat:
    model.apply(set_bn_train)
  tacc=0
  count=0  
  start_time = time.time()
  while(1):  
    inputs,targets=dh.GetBatch()
    inputs=inputs.transpose(1,0,2,3)
    inputs=Variable(torch.from_numpy(inputs).cuda())
    targets=Variable(torch.from_numpy(np.int64(targets)).cuda())
        
    output=model(inputs)
    pred = output.data.max(1)[1] # get the index of the max log-probability
    accuracy = pred.eq(targets.data).cpu().sum().numpy()        
    tacc+=accuracy
    count+=1
    if count==num_batches*eval_fold:
      break
  elapsed = time.time() - start_time
  print ("eval accuracy: ", tacc/(count*targets.data.size(0)+0.0)  )
  #print ('eval time per batch: ', elapsed/(count+0.0))
  return tacc/(count*targets.data.size(0)+0.0)


def test(dh,num_batches,use_bn_trainstat=False, save_cm=False):
  model.eval()
  if use_bn_trainstat:
    model.apply(set_bn_train)
  tacc=0
  count=0  
  start_time = time.time()
  total_testdata=dh.GetDatasetSize()  
  total_ave_acc=np.zeros((total_testdata,outputclass))
  testlabels=np.zeros((total_testdata))
  #print("number of tests: ", test_no)
  cm = np.zeros((outputclass,outputclass)).astype(int)
  if outputclass is 2:
    class_names = np.array(['non-fall', 'fall'])
  else:
    class_names = np.array(range(60))
  while(1):  
    inputs,targets,index=dh.GetBatch()
    inputs=inputs.transpose(1,0,2,3)
    testlabels[index]=targets
    inputs=Variable(torch.from_numpy(inputs).cuda())
    targets=Variable(torch.from_numpy(np.int64(targets)).cuda())
        
    output=model(inputs)
    pred = output.data.max(1)[1] # get the index of the max log-probability
    accuracy = pred.eq(targets.data).cpu().sum().numpy()    
    total_ave_acc[index]+=output.data.cpu().numpy()

    # Plot non-normalized confusion matrix
    cm += get_confusion_matrix(targets.cpu().data.cpu().numpy().astype(int), pred.cpu().data.cpu().numpy().astype(int), classes=class_names)
    
    tacc+=accuracy
    count+=1
    if count==global_test_no*num_batches:
      break    
  #total_ave_acc/=args.test_no
  np.set_printoptions(precision=2)
  plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix, without normalization')
  plt.show()
  
  if save_cm:
    np.save('cm.npy',cm)


  top = np.argmax(total_ave_acc, axis=-1)
  eval_acc=np.mean(np.equal(top, testlabels))    
  elapsed = time.time() - start_time
  print ("test accuracy: ", tacc/(count*targets.data.size(0)+0.0), "eval accuracy: ", eval_acc, "( use_bn_trainstat=", use_bn_trainstat, ")"  )
  #print ('test time per batch: ', elapsed/(count+0.0))
  
  
  return tacc/(count*targets.data.size(0)+0.0)#, eval_acc/(total_testdata+0.0)

def clip_gradient(model, clip):
    for p in model.parameters():
        p.grad.data.clamp_(-clip,clip)
        #print(p.size(),p.grad.data)

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr     

def clip_weight(RNNmodel, clip):
    for name, param in RNNmodel.named_parameters():
      if 'weight_hh' in name:
        param.data.clamp_(-clip,clip)
    
lastacc=0

if outputclass is 60:
  eval_dispFreq=20
else:
  eval_dispFreq=1
eval_dispFreq=20
patience=0
reduced=1
for i in range(1,301):
  print("Epoch: ", i)
  for _ in range(num_train_batches//eval_dispFreq):
    train(eval_dispFreq)
  test_acc=eval(dh_eval,num_eval_batches,use_bneval)

  model_clone = copy.deepcopy(model.state_dict())
  opti_clone = copy.deepcopy(optimizer.state_dict())
  if (test_acc >lastacc):  
    lastacc=test_acc
    patience=0
  elif patience>int(pThre/reduced+0.5):
    reduced=reduced*2
    print ('learning rate',learning_rate)
    model.load_state_dict(model_clone)
    optimizer.load_state_dict(opti_clone)
    patience=0
    learning_rate=learning_rate*0.1
    adjust_learning_rate(optimizer,learning_rate)     
    if learning_rate<end_rate:
      break  
    test_acc=test(dh_test,num_test_batches)     
 
  else:
    patience+=1 
  print('\n')
    
test_acc=test(dh_test,num_test_batches)  
#test_acc=test(dh_test,num_test_batches,True)   