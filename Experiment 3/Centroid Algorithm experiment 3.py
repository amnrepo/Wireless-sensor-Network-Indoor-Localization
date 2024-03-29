import math
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as pb
import random
from datetime import datetime
import time
import sys
import csv



# sys.argv.append('zigbee.csv')  # for local system

sys.argv = ["", "/content/drive/MyDrive/Colab Notebooks/wifi.csv"]    # For google Colab

Anchors = [(0,0),(0,5),(5,0)]


with open(sys.argv[1]) as f:
    dict_from_csv = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]

overall_rss=[]
Tag_nodes=[]

for i in range(len(dict_from_csv)):
    dict=dict_from_csv[i]
    x , y = float(dict['x']) , float(dict['y'])
    Tag_nodes.append((x,y))
    random.seed(datetime.now())

    rss = [-int(float(dict['RSSI A']))-random.random(),-int(float(dict['RSSI B']))-random.random() ,-int(float(dict['RSSI C']))-random.random()]
    overall_rss.append(rss)


if 'wifi' in sys.argv[1]:
    rss0 = -45.73
    pathloss_exponent = 2.162
elif 'ble' in sys.argv[1]:
    rss0 = -75.48
    pathloss_exponent = 2.271 
elif 'zigbee' in sys.argv[1]:
    rss0 = -50.33
    pathloss_exponent = 2.935


def dist_rssi(rss):
    cal_d= pow(10,((rss-rss0)/(-10*pathloss_exponent)))
    return cal_d

def dist(x, y, pos):
    return math.sqrt((pos[0]-x)**2 + (pos[1]-y)**2)

def WC(weight,x,y):    
    
    xiwi=np.multiply(x,weight)
    yiwi=np.multiply(y,weight)
    xw=np.sum(xiwi)/np.sum(weight)
    yw=np.sum(yiwi)/np.sum(weight)
    return xw,yw


distance_error =[]
pp=[]

start_time = time.time()

for i in range(0,len(Tag_nodes)):
    weight = []


    x = overall_rss[i] # Get the RSSI values for the i-th node
    x1 = x[0]   # Assign the first RSSI value
    x2 = x[1]    # Assign the second RSSI value
    x3 = x[2]   # Assign the third RSSI value
       # Assign the fourth RSSI value

    # Calculate the distance between the tag and the first anchor using the RSSI value
    cal_d=dist_rssi(x1)
    # Append the inverse of the distance to the "weight_arr" list
    weight=np.append(weight,(1/cal_d))

    # Calculate the distance between the tag and the second anchor using the RSSI value
    cal_d=dist_rssi(x2)
    # Append the inverse of the distance to the "weight_arr" list
    weight=np.append(weight,(1/cal_d))

    # Calculate the distance between the tag and the third anchor using the RSSI value
    cal_d=dist_rssi(x3)
    weight=np.append(weight,(1/cal_d))

    
    

    
    res_x,res_y=WC(weight,[Anchors[j][0] for j in range(0,3)],[Anchors[j][1] for j in range(0,3)])   # Calling Weighted Centroid Localization (WCL) function for determining the position of a target based on the RSSI (Received Signal Strength Indication) values

    pp.append((res_x,res_y))

    distance_error.append(dist(res_x,res_y,Tag_nodes[i]))




print("Total Time: %s seconds" % (time.time() - start_time))

TError=np.sum(distance_error)/10

MError=np.average(distance_error)/10

stdn=np.std(distance_error)/10

print(" Total Error (Distance): " + str(TError)+"\tMean  Error (Distance): "+str(MError)+"\tStandard Deviation: "+str(stdn))



P =  np.asarray(pp)

O = np.asarray(Tag_nodes)

fig, ax = plt.subplots()



ax.scatter(O[:3,0], O[:3,1], color='g' , marker='o',s=100,label = "Tag" , linewidth=2)

ax.scatter(P[:3,0], P[:3,1], label = "Centroid",color='orange',alpha = 1,marker='s',s=80)

ax.scatter((0,0), (0,4), color='red',label = "Anchor node",marker='D')

ax.scatter(4, 0, color='red',marker='D')



ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Trilateration')

plt.show()