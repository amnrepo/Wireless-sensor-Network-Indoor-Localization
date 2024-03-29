import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from datetime import datetime
import time
import sys
import csv


def calculate_euclidean_distance(x1, y1, x2, y2):
    # calculate the Euclidean distance between two points
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_distance_from_rssi(rssi, rss0, pathloss_exponent):
    # calculate the distance between a node and a receiver based on their RSSI values
    return 10 ** ((rss0 - rssi) / (10 * pathloss_exponent))


def get_candidate_positions(node_positions, distances):
    candidate_positions = []  
    for i in range(len(distances)):  
        candidate_positions_i = []  
        for j in range(3):  # iterate through each node position
            x = node_positions[j][0]  
            y = node_positions[j][1] - distances[i][j]
            sub_location = []
            while y < node_positions[j][1]:  
                x_intermediate = math.sqrt(abs(distances[i][j] ** 2 - (y - node_positions[j][1]) ** 2)) 
               
                sub_location.append((x_intermediate + x, y))
                sub_location.append((x_intermediate + x, -y))
                sub_location.append((-x_intermediate + x, y))
                sub_location.append((-x_intermediate + x, -y))
                y += 1 
            candidate_positions_i.append(sub_location) 
        candidate_positions.append(candidate_positions_i) 
    return candidate_positions  



def compute_error(candidate_positions, original_trajectory, node_positions, distances):
    # This function takes four arguments: candidate_positions, original_trajectory, node_positions, and distances.
    distance_errors = []
    predicted_positions = []
    # Two empty lists are initialized: `distance_errors` and `predicted_positions`.
    for i in range(len(original_trajectory)):
        positions = []
        errors = []
        # Two more empty lists are initialized: `positions` and `errors`.
        for j in range(3):
            for k in range(len(candidate_positions[i][j])):
                position = candidate_positions[i][j][k]
                error = 0
                for l in range(3):
                    error_intermediate = calculate_euclidean_distance(position[0], position[1], node_positions[l][0],
                                                                       node_positions[l][1])
                    error += (error_intermediate - distances[i][l]) ** 2
                errors.append(error)
                positions.append(position)
        min_error = min(errors)
        min_index = errors.index(min_error)
        predicted_position = positions[min_index]
        predicted_positions.append(predicted_position)
        # For each original trajectory point, the function calculates the distance error between the predicted and actual positions.
        distance_errors.append(calculate_euclidean_distance(predicted_position[0], predicted_position[1],
                                                             original_trajectory[i][0], original_trajectory[i][1]))
    # The function returns the distance_errors list and the predicted_positions list.
    return distance_errors, predicted_positions


def main():
    # Define area size and node positions
    area_size = (4, 4)
    node_positions = [(0, 0), (0, 4), (4, 0)]
    
    # Append the filename to sys.argv for command line input
    sys.argv.append('WIFI')  # For local system

    # sys.argv = ["", "/content/drive/MyDrive/Colab Notebooks/wifi.csv"]    # For google Colab

    # Determine RSSI values based on input file type
    if 'wifi' in sys.argv[1]:
        rss0 = -45.73
        pathloss_exponent = 2.162
    elif 'ble' in sys.argv[1]:
        rss0 = -75.48
        pathloss_exponent = 2.271
    elif 'zigbee' in sys.argv[1]:
        rss0 = -50.33
        pathloss_exponent = 2.935
    
    # Read in data from CSV file and convert to dictionary
    with open(sys.argv[1]) as f:
        dict_from_csv = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    
    # Initialize lists to store RSSI and original trajectory data
    overall_rss = []
    original_trajectory = []
    
    # Loop through CSV data and calculate RSSI values
    for i in range(len(dict_from_csv)):
        dict_item = dict_from_csv[i]
        x, y = float(dict_item['x']), float(dict_item['y'])
        original_trajectory.append((x,y))
        random.seed(datetime.now())
        rss = [-int(float(dict_item['RSSI A']))-random.random(),-int(float(dict_item['RSSI B']))-random.random() ,-int(float(dict_item['RSSI C']))-random.random()]
        overall_rss.append(rss)
    
    # Calculate distance between nodes using RSSI values 
    dist_i = []
    for i in range(0,len(overall_rss)):
        dist = []
        for j in range(0,3):
            dist.append(calculate_distance_from_rssi(overall_rss[i][j], rss0, pathloss_exponent))
        dist_i.append(dist)
    
    # Determine candidate positions
    candidate_pos = get_candidate_positions(node_positions, dist_i)
    
    # Compute error between candidate positions and original trajectory
    distance_error, pp = compute_error(candidate_pos, original_trajectory, node_positions, dist_i)

    # Print computation time and distance error statistics
    
    print("Total Time: %s seconds" % (time.time() - start_time))

    TError=np.sum(distance_error)/10

    MError=np.average(distance_error)/10

    stdn=np.std(distance_error)/10

    print(" Total Error (Distance): " + str(TError)+"\tMean  Error (Distance): "+str(MError)+"\tStandard Deviation: "+str(stdn))


    P =  np.asarray(pp)

    O = np.asarray(original_trajectory)
 
    fig, ax = plt.subplots()


    ax.scatter(O[:3,0], O[:3,1], color='g' , marker='o',s=100,label = "Tag" , linewidth=2)

    ax.scatter(P[:3,0], P[:3,1], label = "Trilateration",color='violet',alpha =1,marker='^',s=90)


    ax.scatter((0,0), (0,4), color='red',label = "Anchor node",marker='D')

    ax.scatter(4, 0, color='red',marker='D')


    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title('Trilateration')

    plt.show()




if __name__ == '__main__':
    start_time = time.time()
    main()


