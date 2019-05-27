import os
import json
import numpy as np

duplicate=49
data = []
label = []
test = []
test_label = []
meta = np.load("meta.npy")

seq_len=90

for fall in range(1, 56):
	for cam in range(1,6):
		directory = "./falls_keys/Fall"+str(fall)+"_Cam"+str(cam)+".avi_keys/"
		if os.path.isdir(directory):
			frames = []
			isfall=True

			mid=meta[fall-1][1]*30
			k=mid-seq_len/2
			p=mid+seq_len/2
			
			while k>=0:	
				for i in range(seq_len):
					filename = "Fall"+str(fall)+"_Cam"+str(cam)+"_00000000"+str(k+i).zfill(4)+"_keypoints.json"
					f = open(directory+filename, 'r') 
					datastore = json.load(f)
					if datastore['people']:
						frame = []
						for j in range(0, 54, 3):
							frame.append([datastore['people'][0]['pose_keypoints_2d'][j], datastore['people'][0]['pose_keypoints_2d'][j+1]])
						#print frame
						frames.append(frame)
					f.close()

				if len(frames)!=0:
					#print len(frames)
					#if fall>44:
					if cam==5:
						test.append(frames)
						if isfall:
							test_label.append(1)
							isfall=False
						else:
							test_label.append(0)
					else:
						if isfall:
							for _ in range(duplicate):
								label.append(1)
								data.append(frames)
							isfall=False
						else:
							label.append(0)
							data.append(frames)
					frames=[]
				k-=seq_len

			while p<len(os.listdir(directory)):	
				for i in range(min(seq_len,len(os.listdir(directory))-p)):
					filename = "Fall"+str(fall)+"_Cam"+str(cam)+"_00000000"+str(p+i).zfill(4)+"_keypoints.json"
					f = open(directory+filename, 'r') 
					datastore = json.load(f)
					if datastore['people']:
						frame = []
						for j in range(0, 54, 3):
							frame.append([datastore['people'][0]['pose_keypoints_2d'][j], datastore['people'][0]['pose_keypoints_2d'][j+1]])
						#print frame
						frames.append(frame)
					f.close()

				if len(frames)!=0:
					#uncomment 'if fall...' and comment 'if cam...' for cross-scenario
                    			#cross-view if remain this way
					#if fall>44:
					if cam==5:
						test.append(frames)
						test_label.append(0)
					else:
						label.append(0)
						data.append(frames)
					frames=[]
				p+=seq_len

			print(directory+" finished")
			#print(np.asarray(data).shape)
			#print(np.asarray(label).shape)
			#print label

np.save('train_data_'+str(seq_len)+'.npy',np.asarray(data))
np.save('train_label_'+str(seq_len)+'.npy',np.asarray(label))
np.save('test_data_'+str(seq_len)+'.npy',np.asarray(test))
np.save('test_label_'+str(seq_len)+'.npy',np.asarray(test_label))

