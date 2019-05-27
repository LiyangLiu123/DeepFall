# DeepFall
## Check code on NTU skeleton data in folder NTU.
Download NTU dataset from http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp. Note that permission has to be apllied first.
Modify gen_ndarray.py in NTU folder to generate different numpy arrays to feed in file indrnn.ipynb. You choose btween 2 class and 60 class by saving different numpy array in the bottom of gen_ndarray.py. For example
```
np.save('/Users/liuliyang/Downloads/ndarray/2cls/test_ntus_2cls.npy', data)
np.save('/Users/liuliyang/Downloads/ndarray/2cls/test_ntus_2cls_len.npy', lens)
np.save('/Users/liuliyang/Downloads/ndarray/2cls/test_ntus_2cls_label.npy', targets_2cls)
```
Check code on high quality simulation data in folder high_quality_sim.
