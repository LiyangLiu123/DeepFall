import os
import json
import numpy as np

data = []
label = []
test = []
test_label = []
meta = np.load("meta.npy")

for fall in range(1, 56):
	for cam in range(1,6):
		directory = "./falls_keys/Fall"+str(fall)+"_Cam"+str(cam)+".avi_keys/"
		if os.path.isdir(directory):
			frames = []
			f_count=0
			
			for i in range(len(os.listdir(directory))):
				filename = "Fall"+str(fall)+"_Cam"+str(cam)+"_00000000"+str(i).zfill(4)+"_keypoints.json"
				f = open(directory+filename, 'r') 
				datastore = json.load(f)
				if datastore['people']:
					frame = []
					for j in range(0, 54, 3):
						frame.append([datastore['people'][0]['pose_keypoints_2d'][j], datastore['people'][0]['pose_keypoints_2d'][j+1]])
					#print frame
					frames.append(frame)
					f_count+=1
				f.close()
				if f_count==30:
					f_count=0
					#print len(frames)
					if cam==5:
						test.append(frames)
						if i < meta[fall-1][0]*30 or i > meta[fall-1][2]*30:
							test_label.append(0)
						else:
							test_label.append(1)
					else:
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

np.save('train_data.npy',np.asarray(data))
np.save('train_label.npy',np.asarray(label))
np.save('test_data.npy',np.asarray(test))
np.save('test_label.npy',np.asarray(test_label))

