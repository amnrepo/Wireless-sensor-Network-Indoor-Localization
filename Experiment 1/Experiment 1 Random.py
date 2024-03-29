import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot as pb
from matplotlib.lines import Line2D

import random
from datetime import datetime
import time



Grid=(20, 20)
Anchors = [(-26, 26), (26, 26), (26, -26), (-26, -26)]

Noise =1

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
Tag_Nodes=[]

for i in range(5):

    random.seed(datetime.now())

    x_tag = random.randint(-Grid[0]+2,Grid[0]-2)

    random.seed(datetime.now())

    y_tag = random.randint(-Grid[1]+2,Grid[1]-10)

    Tag_Nodes.append((x_tag,y_tag))
    inter_rss = []
    for j in range(100):
        inter_rss.append(RSSI_ranging(tag=(x_tag,y_tag),noise=2))
    RSSI.append(np.array(np.mean(inter_rss, axis=0)))

   
def dist_rss(rss):
    cal_d= pow(10,((rss-A)/(-10*loss)))
    return cal_d

#-------------------------------------------------------------------
# 1. Multilateration 

dist_i = []
sub_locations = [] 
for i in range(0,len(RSSI)):
    dist_j = []
    for j in range(0,4):
        dist_j.append(dist_rss(RSSI[i][j]))
    dist_i.append(dist_j)
    # print(dist_j)
    sub_locationsc =[]
    
    # The below code performing the Mutlilateration process to determine the tag node's location. 
    for j in range(0,4):  

        x = Anchors[j][0] 

        y = Anchors[j][1]-dist_j[j] 
        
        # print(x,y)
        sub_location = []

        while y < Anchors[j][1]:

            xc = math.sqrt(abs((dist_j[j]**2) - ((y-Anchors[j][1])**2)))

            sub_location.append((xc+x,y))
            
            sub_location.append((xc+x,-y))

            sub_location.append((-xc+x,y))

            sub_location.append((-xc+x,-y))

            y+=1

        sub_locationsc.append(sub_location)

    sub_locations.append(sub_locationsc)

distance_error =[]
LSE_positions = []
start_time = time.time()


for i in range(0,len(Tag_Nodes)):       
    Tag_node =[]
    fxy_list=[]
    for j in range(0,4):    
        for k in range(len(sub_locations[i][j])):
            tag_loc = sub_locations[i][j][k]
            error = 0
            for l in range(0,4):

             
                fxy = math.sqrt(((tag_loc[0]-Anchors[l][0])**2) + ((tag_loc[1]-Anchors[l][1])**2)) 
                
                error = error + math.pow((fxy - dist_i[i][l]),2)

            fxy_list.append(error)


            Tag_node.append(tag_loc)

    min_fxy = min(fxy_list)

    min_index = fxy_list.index(min_fxy)
    Estimated_position = Tag_node[min_index]
    LSE_positions.append(Estimated_position)
    distance_error.append(Euclidean(Estimated_position[0],Estimated_position[1],Tag_Nodes[i]))


print("Total Time: %s seconds" % (time.time() - start_time))

TError=np.sum(distance_error)/10

MError=np.average(distance_error)/10

stdn=np.std(distance_error)/10

print(" Total Error (Distance): " + str(TError)+"\tMean  Error (Distance): "+str(MError)+"\tStandard Deviation: "+str(stdn))



#-----------------------------------------------

# 2. Centroid Algorithm 

def WC(weight,x,y):    
    
    xiwi=np.multiply(x,weight)
    yiwi=np.multiply(y,weight)
    xw=np.sum(xiwi)/np.sum(weight)
    yw=np.sum(yiwi)/np.sum(weight)
    return xw,yw


# def wcl(weight,x,y):
#     xiwi=np.multiply(x,weight)
#     yiwi=np.multiply(y,weight)
#     xw=np.sum(xiwi)/np.sum(weight)
#     yw=np.sum(yiwi)/np.sum(weight)
#     return xw,yw

distance_error =[]

start_time = time.time()

WCPosistions = []


for i in range(0,len(Tag_Nodes)):
    weight_arr = []
    x = RSSI[i]
    x1 = x[0]   # Assign the first RSSI value
    x2 = x[1]    # Assign the second RSSI value
    x3 = x[2]   # Assign the third RSSI value
    x4 = x[3]   # Assign the fourth RSSI value

    # Calculate the distance between the tag and the first anchor using the RSSI value
    cal_d=dist_rss(x1)
    # Append the inverse of the distance to the "weight_arr" list
    weight_arr=np.append(weight_arr,(1/cal_d))

    # Calculate the distance between the tag and the second anchor using the RSSI value
    cal_d=dist_rss(x2)
    weight_arr=np.append(weight_arr,(1/cal_d))

    cal_d=dist_rss(x3)
    weight_arr=np.append(weight_arr,(1/cal_d))

    cal_d=dist_rss(x4)
    weight_arr=np.append(weight_arr,(1/cal_d))

    res_x,res_y=WC(weight_arr,[Anchors[j][0] for j in range(0,4)],[Anchors[j][1] for j in range(0,4)])      # Calling Weighted Centroid Localization (WCL) function for determining the position of a target based on the RSSI (Received Signal Strength Indication) values
    WCPosistions.append((res_x,res_y))
    distance_error.append(Euclidean(res_x,res_y,Tag_Nodes[i]))

print("Total Time: %s seconds" % (time.time() - start_time))

TError=np.sum(distance_error)/10

MError=np.average(distance_error)/10

stdn=np.std(distance_error)/10

print(" Total Error (Distance): " + str(TError)+"\tMean  Error (Distance): "+str(MError)+"\tStandard Deviation: "+str(stdn))



#-----------------------------------------------------------------------

# 3.  Grid RSS


previous_errors =[]
distance_error =[]
particles = []
times = []
differential_positions = []
start_time = time.time()
Limit = list(range(-20, 20))
num_particles = 1000


for x in range(num_particles):
    random.seed(datetime.now())
    particles.append((random.choice(Limit),random.choice(Limit)))
for i in range(0,len(Tag_Nodes)):
    positions =[]
    errors=[]
    weights =[]
    rands = []
    range_probs = []
    error=0
    for particle in particles:
        x,y=particle[0],particle[1]
        actual_rss = RSSI_ranging(tag=(x,y),noise=0)
        error=np.sum(np.subtract(actual_rss,RSSI[i]))
        if previous_errors:
            std_error=np.std(previous_errors)
        else:
            std_error=0.001
        
        std_error=np.std(np.subtract(actual_rss,RSSI[i]))
        omega=((1/((4*std_error)*math.sqrt(2*math.pi)))*(math.pow(math.e,-(math.pow(error,2)/2*np.square(4*std_error)))))
        for j in range(len(previous_errors)-1,len(previous_errors)-4 if len(previous_errors) > 5 else 0,-1):
            omega=omega*((1/((4*std_error)*math.sqrt(2*math.pi)))*(math.pow(math.e,-(math.pow(previous_errors[j],2)/2*np.square(4*std_error)))))
    
        weights.append(omega)
        positions.append((x,y))
        errors.append(error)
        
    sum_weight=np.sum(weights)
    if sum_weight == 0:
        pass
    for j in range(0,len(weights)):
        weights[j]=weights[j]/sum_weight


    max_weight = max(weights)
    max_index = weights.index(max_weight)
    pos=positions[max_index]
    previous_errors.append(errors[max_index])
    differential_positions.append(pos)
    distance_error.append(Euclidean(pos[0],pos[1],Tag_Nodes[i]))



print("Total Time: %s seconds ---" % (time.time() - start_time))

TError=np.sum(distance_error)/10

MError=np.average(distance_error)/10

stdn=np.std(distance_error)/10

print(" Total Error (Distance): " + str(TError)+"\tMean  Error (Distance): "+str(MError)+"\tStandard Deviation: "+str(stdn))





plt.title("Localization (Randomly distributed tag nodes)")
plt.scatter(np.array(Tag_Nodes)[:,0],np.array(Tag_Nodes)[:,1] , marker='o',s=100,alpha = 0.5, color='green',label='Tag nodes')

plt.scatter(np.array(WCPosistions)[:,0],np.array(WCPosistions)[:,1],marker='s',s=80, color='pink',label='Centroid')

plt.scatter(np.array(LSE_positions)[:,0],np.array(LSE_positions)[:,1],marker='^',s=90, color='blue',label='Multilateration')

plt.scatter(np.array(differential_positions)[:,0],np.array(differential_positions)[:,1],marker='x',s=80, color='orange',label='Grid RSS')






plt.legend()
plt.show(block=False)
plt.ioff()
plt.show()


