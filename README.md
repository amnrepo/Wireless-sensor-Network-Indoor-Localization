Abstract—This project aims to compare the performance of three localization techniques: Trilateration/Multilateration, Centroid algorithm, and Grid-based RSS, for indoor node localization in wireless networking applications using Received Signal Strength Indication (RSSI) data. Three experiments were conducted, each utilizing an internally generated or real-world dataset with RSSI values for an unknown or tag node. The datasets were obtained from various sources and used in different scenarios to evaluate the performance of the three techniques. The results of the experiments were analyzed and compared using various metrics, such as mean error, standard deviation, and computation time. The findings suggest that the Trilateration technique outperformed the other techniques in terms of accuracy and precision in Bluetooth environments, while the Centroid technique demonstrated the highest robustness to noise and outliers. These results can assist researchers and practitioners in selecting the most suitable localization technique for their specific wireless networking application based on the specific requirements and constraints.
Keywords—Indoor localization, RSSI, Mutlilateration, Grid Based RSS, Wireless Sensor Network, performance evaluation.


# Wireless-sensor-Network-Indoor-Localization

Refer **Project Paper 2.docx** for the project information.

NOTE:- Use Google colab to run the Experiments/code. 

#Experiment 1 - This folder contains single .py file for experiment 1.

#Experiment 2 - This folder contains .py files for experiment 2.

#Experiment 3 - This folder contains .py files for experiment 3. The folder also contains dataset required by .py files to run the code. Use 

Below command to run the "Experiment 3" with desired dataset on google colab.

	sys.argv = ["", "/content/drive/MyDrive/Colab Notebooks/wifi.csv"]

To run "Experiment 3" on local system like in vscode, use below command.

	sys.argv.append('WIFI.csv')
