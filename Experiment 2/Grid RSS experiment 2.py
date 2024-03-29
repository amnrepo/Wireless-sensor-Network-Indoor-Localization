import math
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as pb
import random
from datetime import datetime
import time

# distance Calculation
def dist(x, y, pos):
    return math.sqrt((pos[0]-x)**2 + (pos[1]-y)**2)

# Simulation Environment variable Initialization
areaSize=(30, 30)
node_positions = (areaSize[0]+6,areaSize[1]+6)
node_pos=[(-node_positions[0],node_positions[1]),(node_positions[0],node_positions[1]),(node_positions[0],-node_positions[1]),(-node_positions[0],-node_positions[1])]
initial_pos=(0,0) 
possible_value = list(range(-30, 30))

num_particles = 200
NOISE_LEVEL=1
RESOLUTION=5
Delta_X = 72
Delta_Y = 72
STEP_SIZE=10

# RSSI signal generation at pos(x,y) using path-loos model

def gen_wifi(freq=2.4, power=20, trans_gain=0, recv_gain=0, size=areaSize, pos=(5,5), shadow_dev=2, n=3,noise=NOISE_LEVEL):
    if pos is None:
        pos = (random.randrange(size[0]), random.randrange(size[1]))

    random.seed(datetime.now())
    rss0 = power + trans_gain + recv_gain + 20 * math.log10(3 / (4 * math.pi * freq * 10))
    rss0=rss0-noise*random.random()
    normal_dist = np.random.normal(0, shadow_dev, size=[size[0]+1, size[1]+1])
    rss = []
    random.seed(datetime.now())

    for x in range(0,4):
    # for x in range(0,3):
        distance = dist(node_pos[x][0], node_pos[x][1], pos)
        val =rss0 - 10 * n * math.log10(distance) + normal_dist[int(pos[0])][int(pos[1])]
        rss.append(val-noise*random.random())
    return rss

text=[]
overall_rss=[]
original_tragectory=[]
Previous_pos = initial_pos

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
    rss = gen_wifi(pos=(x,y))
    overall_rss.append(rss)

pp=[]
previous_errors =[]
distance_error =[]
particles = []
times = []
differential_positions = []
start_time = time.time()

for x in range(num_particles):
    random.seed(datetime.now())
    particles.append((random.choice(possible_value),random.choice(possible_value)))
    
for i in range(0,len(OT)):
    positions =[]
    errors=[]
    weights =[]
    rands = []
    range_probs = []
    error=0
    for particle in particles:
        x,y=particle[0],particle[1]
        actual_rss = gen_wifi(pos=(x,y),noise=0)
        error=np.sum(np.subtract(actual_rss,overall_rss[i]))
        if previous_errors:
            std_error=np.std(previous_errors)
        else:
            std_error=0.001
        
        std_error=np.std(np.subtract(actual_rss,overall_rss[i]))
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
    pp.append(pos)
    differential_positions.append(pos)
    distance_error.append(dist(pos[0],pos[1],OT[i]))


distcumulativeEror=np.sum(distance_error)/10
distmeanError=np.average(distance_error)/10
distStandardDeviationError=np.std(distance_error)/10

print("***Total Time: %.2f seconds" % (time.time() - start_time))
print("***Total Error: %.2f" % distcumulativeEror)
print("***Mean Error: %.2f" % distmeanError)
print("***Standard Deviation: %.2f\n" % distStandardDeviationError)

resultFile = open("error_boundry_Markov_full(6by6).csv", "a")  
resultFile.write(str(distcumulativeEror)+","+str(distmeanError)+","+str(distStandardDeviationError)+"\n")
resultFile.close()

O=OT
P =  np.asarray(pp)

# Create a figure and axis object
fig, ax = plt.subplots()
ax.scatter(O[:,0], O[:,1], color='green' , alpha=0.5,label = "Tag nodes" , linewidth=2)
ax.plot(O[:,0], O[:,1], color='green', alpha=0.1, linewidth=1)
ax.plot(P[:,0], P[:,1], color=(0.5, 0.5, 0.5, 0.1),marker='o',markerfacecolor='blue',label = "Estimate", linewidth=1)

for i, (xi, yi) in enumerate(zip(O[:,0], O[:,1])):
    ax.text(xi, yi, str(i+1), color='green', fontsize=10)

for pi, (pxi, pyi) in enumerate(zip(P[:,0], P[:,1])):
    ax.text(pxi, pyi, str(pi+1), color='blue', fontsize=10)   

ax.scatter(node_pos[0], node_pos[1], color='Red',label = "Anchors",marker='D')
ax.scatter(node_pos[2], node_pos[3], color='Red',marker='D')

# Add labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Grid RSS (Boundry)')
plt.legend(prop={'size': 7})
# Display the plot
plt.ioff()
plt.show()
