from matplotlib import pyplot as pb
import random
from datetime import datetime
import time
import math
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D






Anchors = [(-36, 36), (36, 36), (36, -36), (-36, -36)]
Noise=1
loss = 3
random.seed(datetime.now())

A = 20 + 20 * math.log10(3 / (4 * math.pi * 2.4 * 10))   #Friis transmission equation to calculate signal power at one meter  . Using the values for frequency, path loss exponent (n), transmitted power, and antenna gains which are commonly/widely accepted.

A = A - Noise*random.random()


def Euclidean(x, y, tag):
    return math.sqrt((tag[0]-x)**2 + (tag[1]-y)**2)

def RSSI_ranging(tag=(5,5),loss=3,noise=Noise):
    
    PWR = []

    for x in range(0,4):

        distance = Euclidean(Anchors[x][0], Anchors[x][1], tag) # Calculate distance between anchor node and position

        pwr = A - 10 * loss * math.log10(distance) 

        pwr = pwr-noise*random.random()

        PWR.append(pwr)

    return PWR


RSSI=[]


# Distribution of tag nodes
OT =[(0, 0),
 (0, 20),
 (0, 30),
 (20, 30),
 (30, 30),
 (30, 10),
 (30, -10),
 (30, -30),
 (10, -30),
 (-10, -30),
 (-30, -30),
 (-30, -10),
 (-30, 10),
 (-30, 30),
 (-10, 30)]

OT = np.asarray(OT)
OTx = OT[:,0]
OTy = OT[:,1]

for z in range(0,len(OT)):
    x,y = OTx[z], OTy[z]
    rss = RSSI_ranging(tag=(x,y))
    RSSI.append(rss)


n=3
a=A
def dist_rssi(rss,a,n):
    cal_d= pow(10,((rss-a)/(-10*n)))
    return cal_d


def WC(weight,x,y):    
    
    xiwi=np.multiply(x,weight)
    yiwi=np.multiply(y,weight)
    xw=np.sum(xiwi)/np.sum(weight)
    yw=np.sum(yiwi)/np.sum(weight)
    return xw,yw

distance_error =[]

pp=[]

start_time = time.time()

# plt.title("Predicted Robot Positions")
for i in range(0,len(OT)):
    weight = []


    x = RSSI[i] # Get the RSSI values for the i-th node
    x1 = x[0]   # Assign the first RSSI value
    x2 = x[1]    # Assign the second RSSI value
    x3 = x[2]   # Assign the third RSSI value
    x4 = x[3]   # Assign the fourth RSSI value

    # Calculate the distance between the tag and the first anchor using the RSSI value
    cal_d=dist_rssi(x1,a,n)
    # Append the inverse of the distance to the "weight_arr" list
    weight=np.append(weight,(1/cal_d))

    # Calculate the distance between the tag and the second anchor using the RSSI value
    cal_d=dist_rssi(x2,a,n)
    # Append the inverse of the distance to the "weight_arr" list
    weight=np.append(weight,(1/cal_d))

    # Calculate the distance between the tag and the third anchor using the RSSI value
    cal_d=dist_rssi(x3,a,n)
    weight=np.append(weight,(1/cal_d))

    # Calculate the distance between the tag and the fourth anchor using the RSSI value
    cal_d=dist_rssi(x4,a,n)
    weight=np.append(weight,(1/cal_d))

    
    res_x,res_y=WC(weight,[Anchors[j][0] for j in range(0,4)],[Anchors[j][1] for j in range(0,4)])   # Calling Weighted Centroid Localization (WCL) function for determining the position of a target based on the RSSI (Received Signal Strength Indication) values

    pp.append((res_x,res_y))

    distance_error.append(Euclidean(res_x,res_y,OT[i]))


print("Total Time: %s seconds ---" % (time.time() - start_time))

TError=np.sum(distance_error)/10

MError=np.average(distance_error)/10

stdn=np.std(distance_error)/10

print(" Total Error (Distance): " + str(TError)+"\tMean  Error (Distance): "+str(MError)+"\tStandard Deviation: "+str(stdn))

O = OT
P =  np.asarray(pp)

fig, ax = plt.subplots()


ax.scatter(O[:,0], O[:,1], color='green' , alpha=0.5,label = "Tag nodes" , linewidth=2)

ax.plot(O[:,0], O[:,1], color='green', alpha=0.1, linewidth=1)

ax.plot(P[:,0], P[:,1], color=(0.5, 0.5, 0.5, 0.1),marker='o',markerfacecolor='blue',label = "Estimate", linewidth=1)

for i, (xi, yi) in enumerate(zip(O[:,0], O[:,1])):
    ax.text(xi, yi, str(i+1), color='green', fontsize=10)

for pi, (pxi, pyi) in enumerate(zip(P[:,0], P[:,1])):
    ax.text(pxi, pyi, str(pi+1), color='blue', fontsize=10)   

ax.scatter(Anchors[0], Anchors[1], color='Red',label = "Anchors",marker='D')

ax.scatter(Anchors[2], Anchors[3], color='Red',marker='D')


# Add labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Centroid (Boundry)')

plt.legend(prop={'size': 6.5})
# Display the plot
plt.ioff()
plt.show()

