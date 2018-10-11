The controller runs using:
python2 controller.py "config_file"

example: python2 controller.py config_south.yml

To write your own config file use "config_south.yml" as an example

File explanations:

controller.py : The main script for using the shortest path algorithm in the ciee building

DataManager.py : This script contains the datamanager class, which is responsible for downloading all the required data
for the system

Advise.py : Contains the advise class. Initializes all the required models, runs the shortest path algorithm and returns
the control action

EnergyConsumption.py, ThermalModel.py, Discomfort.py, Occupancy.py: Files that contain all the required models for
the shortest path.

utils.py : contain graph utility methods

Safety.py : Contains the safety class, which is responsible for the available actions for the algorithm, and all the
checks needed to ensure the safety constraints are met


TODO:

* Historical weather forecasting reports (needed to integrate the evaluation)
* Confirm the integrity of the Brick queries
* Pricing, occupancy, setpoints from CMU and UI respectively
* Manual Tstat controls (what to do with them? store them?)
