# DeepFall
## The original code
The original version code is available at https://github.com/Sunnydreamrain/IndRNN_pytorch.  
More details can be found there.
## Check code on NTU skeleton data in folder NTU.
Download NTU dataset from http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp. Note that permission has to be apllied first.
Modify gen_ndarray.py in NTU folder to generate different numpy arrays to feed in file indrnn.ipynb. You choose btween 2 class and 60 class by saving different numpy array in the bottom of gen_ndarray.py. For example
```
np.save('/Users/liuliyang/Downloads/ndarray/2cls/test_ntus_2cls.npy', data)
np.save('/Users/liuliyang/Downloads/ndarray/2cls/test_ntus_2cls_len.npy', lens)
np.save('/Users/liuliyang/Downloads/ndarray/2cls/test_ntus_2cls_label.npy', targets_2cls)
```
## Check code on high quality fall simulation data in folder high_quality_sim.
You have to download the videos from https://iiw.kuleuven.be/onderzoek/advise/datasets#High%20Quality%20Fall%20Simulation%20Data and use OpenPose to transform the videos into COCO key points with 18 joints before using the codes here.

Use process.py and gen_len.py to generate necessary numpy arrays to feed in high_quality_simulation.ipynb. Modify process.py to generate cross-scenario or cross-view evaluation. After every modification you have to run both process.py and gen_len.py again.

Training logs of the author is attached as a zip file. After unzipping, you will find several RTFD files which can be open by the TextEdit app of mac os. You will find training logs and confusion matrices inside each RTFD file.
## About Google Colab
Note that before running in Google Colab, check that under Runtime->change runtime type, runtime type set as Python 2 and hardware accelerator as GPU.
