import math
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as pb
import random
from datetime import datetime
import time




Anchors = [(-36, 36), (36, 36), (36, -36), (-36, -36)]
Noise=1
loss = 3
random.seed(datetime.now())

#Friis transmission equation to calculate signal power at one meter  . Using the values for frequency, path loss exponent (n), transmitted power, and antenna gains which are commonly/widely accepted.
A = 20 + 20 * math.log10(3 / (4 * math.pi * 2.4 * 10))   

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


text=[]
overall_rss=[]
original_tragectory=[]
# Previous_pos = initial_pos
    
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
    overall_rss.append(rss)




n=3

def getDistanceFromRSS(rssi):
    return math.pow(10,((A-rssi)/(10*n)))

dist_i = []
candidate_pos = []
for i in range(0,len(overall_rss)):
    dist_j = []
    for j in range(0,4):
        dist_j.append(getDistanceFromRSS(overall_rss[i][j]))
    dist_i.append(dist_j)
    # print(dist_j)
    candidate_pos_j =[]

    # The below code performing the Mutlilateration process to determine the tag node's location.

    for j in range(0,4):
        y = Anchors[j][1]-dist_j[j]
        x = Anchors[j][0]
        # print(x,y)
        sub_location = []
        while y < Anchors[j][1]:
            xc = math.sqrt(abs((dist_j[j]**2) - ((y-Anchors[j][1])**2)))
            sub_location.append((xc+x,y))

            sub_location.append((xc+x,-y))

            sub_location.append((-xc+x,y))

            sub_location.append((-xc+x,-y))
            y+=1

        candidate_pos_j.append(sub_location)

    candidate_pos.append(candidate_pos_j)


distance_error =[]
start_time = time.time()
pp=[]

for i in range(0,len(OT)):
    positions =[]
    errors=[]
    for j in range(0,4):
        for k in range(len(candidate_pos[i][j])):
            position = candidate_pos[i][j][k]
            error = 0
            for l in range(0,4):

                   #non-linear least squares technique to minimize the positing error
                error_inter = math.sqrt(((position[0]-Anchors[l][0])**2) + ((position[1]-Anchors[l][1])**2))
                error = error + math.pow((error_inter - dist_i[i][l]),2)

            errors.append(error)

            positions.append(position)
    min_error = min(errors)
    min_index = errors.index(min_error)
    predicted_pos = positions[min_index]
    pp.append(predicted_pos)
    distance_error.append(Euclidean(predicted_pos[0],predicted_pos[1],OT[i]))
    
print("Total Time: %s seconds ---" % (time.time() - start_time))

TError=np.sum(distance_error)/10

MError=np.average(distance_error)/10

stdn=np.std(distance_error)/10

print(" Total Error (Distance): " + str(TError)+"\tMean  Error (Distance): "+str(MError)+"\tStandard Deviation: "+str(stdn))



O = OT


P =  np.asarray(pp)



fig, ax = plt.subplots()

# Create a line plot
ax.scatter(O[:,0], O[:,1], color='green' , alpha=0.5,label = "Tag nodes" , linewidth=2)

ax.plot(O[:,0], O[:,1], color='green', alpha=0.1, linewidth=1)

ax.plot(P[:,0], P[:,1], color=(0.5, 0.5, 0.5, 0.1),marker='o',markerfacecolor='blue',label = "Estimate", linewidth=1)

for i, (xi, yi) in enumerate(zip(O[:,0], O[:,1])):
    ax.text(xi, yi, str(i+1), color='green', fontsize=10)

for pi, (pxi, pyi) in enumerate(zip(P[:,0], P[:,1])):
    ax.text(pxi, pyi, str(pi+1), color='blue', fontsize=10)   

ax.scatter(Anchors[0], Anchors[1], color='Red',label = "Anchors",marker='D')

ax.scatter(Anchors[2], Anchors[3], color='Red',marker='D')



ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('MultiLateration')
plt.legend(prop={'size': 6.5})

plt.ioff()
plt.show()