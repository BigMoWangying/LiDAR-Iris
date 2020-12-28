import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import math

from functools import reduce
 
def str2int(s):
    def fn(x,y):
        return x*10+y
    def char2num(s):
        return {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}[s]
    return reduce(fn,map(char2num,s))

sys.path.append("..")

seq = "05"
N = 0


#read kitti pose
traj = np.loadtxt(seq +".txt")

# ./demo [lidar iris result]
dist0 = np.loadtxt("./test_res"+seq+".txt")
gt = {}
for line in open("./gt"+seq+".txt", "r"):
    if line.strip():
        sl = line.split()
        if len(sl) >= 2:
            if str2int(sl[0]) - str2int(sl[1]) > 300:
                gt[sl[0]] = 1
                N = N+1
            else:
                gt[sl[0]] = 0

print(N)

x_cord = traj[:,3]
z_cord = traj[:,11]

#dist 0
th0 = []
pre0 = []
rec0 = []
dist0_min = dist0[:, 2].min()
dist0_max = dist0[:, 2].max()

print(dist0_min, dist0_max)
for i in np.arange(dist0_min, dist0_max + (dist0_max-dist0_min) * 1.0 /50, (dist0_max-dist0_min) * 1.0 /50):
    print(i)
    tp = 0
    p = 0
    for j in range(0, dist0.shape[0]):
        if dist0[j][2] <= i:
            p = p+1
            if dist0[j][3] == 1.0:
                tp = tp+1
    
    re = tp * 1.0 / N
    pr = tp * 1.0 / p
    th0.append(i)
    rec0.append(re)
    pre0.append(pr)

thres = 0
for i in range(len(th0)-1):
    print([th0[i],pre0[i],rec0[i]])
    if pre0[i]==1.0 and pre0[i+1]!=1.0:
        thres = th0[i]
        break 


##draw p-r curve
#coding:utf-8
fig1 = plt.figure(1) # create figure 1
plt.title('Precision/Recall Curve',fontsize=20)# give plot a title
plt.xlabel('Recall', fontsize=20)# make axis labels
plt.ylabel('Precision',fontsize=20)
plt.tick_params(labelsize=18)
plt.plot(rec0, pre0,  "r", label = "LiDAR Iris", linewidth=3.0)
plt.legend(loc="lower left", fontsize=20)


fig2 = plt.figure(2)
plt.title('trajectory',fontsize=20)# give plot a title
plt.xlabel('x', fontsize=20)# make axis labels
plt.ylabel('z',fontsize=20)
plt.tick_params(labelsize=18)
plt.plot(x_cord, z_cord,  "k", linewidth=1.0)

for i in range(len(dist0[:,0])):
    if gt[str(int(dist0[i][0]))]:
        index = int(int(dist0[i][0])-1)
        plt.scatter(x_cord[index], z_cord[index], c="g",alpha=0.2)
    if dist0[i][2] <= thres and dist0[i][3] == 1:
        index = int(dist0[i][0]-1)
        plt.scatter(x_cord[index], z_cord[index], c="r")
    if dist0[i][2] <= thres and dist0[i][3] == 0:
        index = int(dist0[i][0]-1)
        plt.scatter(x_cord[index], z_cord[index], c="b")

plt.show()
