import numpy as np

f = open("meta.txt","r")
lines = f.readlines()
res = []
for line in lines:
	start_min = int(line[3:5])
	start_sec = int(line[6:8])
	fall_min = int(line[15:17])
	fall_sec = int(line[18:20])
	end_min = int(line[27:29])
	end_sec = int(line[30:32])

	start = start_min*60+start_sec
	fall = fall_min*60+fall_sec
	end = end_min*60+end_sec

	res.append([start, fall, end])

print(res)
np.save("meta.npy",res)