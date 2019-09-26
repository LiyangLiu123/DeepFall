import sys
import h5py
import numpy as np
import time
import random
import pickle
from multiprocessing import Pool
from threading import Thread
import os.path

from train_indrnn import test_dataset

datasets=test_dataset
class batch_thread():
  def __init__(self, result, batch_size_,seq_len):#, datasets
    self.result = result
    self.batch_size_=batch_size_
    self.datasets = datasets   
    self.seq_len=seq_len
    self.idx=-1
    
    dataname=datasets+'.npy'
    labelname=datasets+'_label.npy'
    lenname=datasets+'_len.npy'
    self.data_handle=np.load(dataname, allow_pickle=True)
    self.label_handle=np.load(labelname, allow_pickle=True)
    self.len_handle=np.load(lenname, allow_pickle=True) 
    
    self.num_videos = len(self.data_handle)    
    self.shufflevideolist=np.arange(self.num_videos)
    np.random.shuffle(self.shufflevideolist)

    print ('Dataset test size', self.num_videos)
  
  def __call__(self):###Be careful.  The appended data may change like pointer.
    templabel=[] 
    batch_data=[]
    tempindex=[] 
    for j in range(self.batch_size_):
      self.idx +=1
      if self.idx == self.num_videos:
        self.idx =0
        np.random.shuffle(self.shufflevideolist)
      shufflevideoindex=self.shufflevideolist[self.idx]
      
      label=self.label_handle[shufflevideoindex]     
      templabel.append(np.int32(label))  
      tempindex.append(np.int32(shufflevideoindex)) 
      dataset=self.data_handle[shufflevideoindex]
      len_data=self.len_handle[shufflevideoindex] 
      
      #dataset = np.array(dataset)
      
      sample=np.zeros(tuple((self.seq_len,)+dataset.shape[1:]))
      lenperseg=len_data//self.seq_len
      if lenperseg==1 and len_data>self.seq_len:
        startid=np.random.randint(len_data-self.seq_len)
        sample=dataset[startid:startid+self.seq_len]
      elif len_data<=self.seq_len:
        startid=np.random.randint(max(self.seq_len-len_data,int(0.25*self.seq_len)))
        endid=min(self.seq_len,startid+len_data)
        datasid=0
        dataeid=len_data
        if startid+len_data>self.seq_len:
          datasid=np.random.randint(startid+len_data-self.seq_len)
          dataeid=datasid+self.seq_len-startid
        sample[startid:endid]=dataset[datasid:dataeid]
      else:      
        for framei in range(self.seq_len):        
          if framei==self.seq_len-1:
            index=lenperseg*framei + np.random.randint(len_data-lenperseg*(self.seq_len-1))
          else:
            index=lenperseg*framei + np.random.randint(lenperseg)    
          sample[framei]=dataset[index]
        #print (index,lenperseg)  
        
      batch_data.append(sample) ###Be careful. It has to be different. Otherwise, the appended data will change as well.
      #print(batch_data)       

       
    self.result['data']=np.asarray(batch_data,dtype=np.float32)
    self.result['label']= np.asarray(templabel,dtype=np.int32)   
    self.result['index']= np.asarray(tempindex,dtype=np.int32)   
      
      
  def GetDatasetSize(self):
    return self.num_videos



class testDataHandler(object):

  def __init__(self, batch_size, seq_len):#, datasets
    self.batch_size_ = batch_size		
    #self.datasets = datasets    
    random.seed(10)  
    
    self.thread_result = {}
    self.thread = None

    self.batch_advancer =batch_thread(self.thread_result,self.batch_size_,seq_len)#, self.datasets
    
    self.datasetsize=self.batch_advancer.GetDatasetSize()
    
    self.dispatch_worker()
    self.join_worker()


  def GetBatch(self):
    #self.batch_data_  = np.zeros((self.batch_size_, 3, self.seq_length_, 112, 112), dtype=np.float32)
    if self.thread is not None:
      self.join_worker() 
      
#     self.batch_data_=self.thread_result['data']
#     self.batch_label_=self.thread_result['label']

    self.batch_data_=self.thread_result['data']
    self.batch_label_= self.thread_result['label']
    self.batch_index_= self.thread_result['index']
        
    self.dispatch_worker()
    return self.batch_data_, self.batch_label_,self.batch_index_


    


  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None
    
  def GetDatasetSize(self):
    return self.datasetsize