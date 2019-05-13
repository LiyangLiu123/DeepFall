
import os
import json
import numpy as np

data = []
label = []
meta = np.load("meta.npy")

for fall in range(1, 56):
	for cam in range(1,6):
		directory = "./falls_keys/Fall"+str(fall)+"_Cam"+str(cam)+".avi_keys/"
		if os.path.isdir(directory):
			frames = []
			
			for i in range(len(os.listdir(directory))):
				filename = "Fall"+str(fall)+"_Cam"+str(cam)+"_00000000"+str(i).zfill(4)+"_keypoints.json"
				f = open(directory+filename, 'r') 
				datastore = json.load(f)
				if datastore['people']:
					frames.append(datastore['people'][0]['pose_keypoints_2d'])
				f.close()
				if (i+1)%30 is 0:
					if len(frames) is 30:
						
						if i < meta[fall-1][0]*30 or i > meta[fall-1][2]*30:
							label.append(0)
							data.append(frames)
						else:
							for _ in range(50):
								label.append(1)
								data.append(frames)
					frames=[]

			print(directory+" finished")
			#print(np.asarray(data).shape)
			#print(np.asarray(label).shape)
			#print label

np.save('data.npy',np.asarray(data))
np.save('label.npy',np.asarray(label))
