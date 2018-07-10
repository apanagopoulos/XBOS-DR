import os
import yaml

# Go to folder where all config files are located
config_folder_location = "../Server/Buildings/"
all_dir = os.walk(config_folder_location).next()
# Go through all directories in the buildings folder
for directory in all_dir[1]:
    print("========================")
    end_dir = config_folder_location + directory + "/ZoneConfigs"
    if not os.path.isdir(end_dir):
        print("%s has not ZoneConfigs folder. Moving on to next folder" % (config_folder_location + directory))
        continue

    building_config_name = config_folder_location + directory + "/" + directory + ".yml"
    if not os.path.isfile(building_config_name):
        print("Warning: directory %s has no building config %s." % (directory, directory+".yml"))

    # ===== Work on building config here =====
    print("Working on building config file %s" % directory)
    # open config
    with open(building_config_name, 'r') as f:
        building_config = yaml.load(f)

    # # Get ready for DR-event so we can run on server
    # building_config["Server"] = False
    #
    # # Setting DR start and end
    # building_config["Pricing"]["DR_Start"] = "14:00"
    # building_config["Pricing"]["DR_Finish"] = "18:00"

    building_config["Server"] = True
    if "Agent_IP" not in building_config:
        building_config["Agent_IP"] = "172.17.0.1:28589"

    if "Entity_File" not in building_config:
        building_config["Entity_File"] = "./thanos.ent"


    if "Interval_Length" not in building_config:
        building_config["Interval_Length"] = 15
    if "Max_Actions" not in building_config:
        building_config["Max_Actions"] = 400

    building_config["Pricing"]["DR_Start"] = "14:00"
    building_config["Pricing"]["DR_Finish"] = "18:00"

    building_config["Pricing"].pop("DR Start", None)
    building_config["Pricing"].pop("DR Finish", None)
    building_config["Pricing"].pop("DR-Start", None)
    building_config["Pricing"].pop("DR_End", None)


    building_config["Pricing"]["DR"] = True

    # write to config
    with open(building_config_name, 'wb') as f:
        yaml.dump(building_config, f)

    # ===== End building config =====

    files = os.walk(end_dir).next()[2]
    print("in dir: ", end_dir)
    for f in files:
        if ".yml" not in f:
            print("%s is not a yaml file. Continue to next file." % f)
            continue
        print("In zone %s" % f)
        # Loads the configs
        with open("./" + end_dir + "/" + f, 'r') as o:
            config = yaml.load(o)

        # ============ Zone config ==================

        # Set lambdas
        # config["Advise"]["DR_Lambda"] = 0.7
        # config["Advise"]["General_Lambda"] = 0.995
        #
        # # Decide whether to run MPC.
        # if "Cooling_Consumption_Stage_2" in config["Advise"] and not directory == "south-berkeley-senior-center": # we will test thermal model on a two stage cooling building.
        #     config["Advise"]["MPC"] = False
        #     print("%s zone will NOT run MPC." % f)
        # else:
        #     config["Advise"]["MPC"] = True
        #     print("%s zone will run MPC." % f)
        config["Advise"]["Baseline_Dr_Extend_Percent"] = 4
        config["Advise"]["General_Lambda"] = 0.995
        config["Advise"]["DR_Lambda"] = 0.995
        config["Advise"].pop("Lambda", None)

        config.pop("General_Lambda", None)
        config.pop("DR_Lambda", None)


        config["Advise"]["Actuate"] = True
        if building_config["Building"] in ["avenal-public-works-yard", "avenal-recreation-center"]:
            config["Advise"]["Actuate_Start"] = "15:00"
            config["Advise"]["Actuate_End"] = "00:00"
        else:
            config["Advise"]["Actuate_Start"] = "08:00"
            config["Advise"]["Actuate_End"] = "00:00"
        if building_config["Building"] == "jesse-turner-center" and "Basketball" not in config["Zone"]:
            config["Advise"]["Actuate"] = False



        if "Stage_2_Cooling" in config["Advise"]:
            if config["Advise"]["Stage_2_Cooling"]:
                config["Advise"]["MPC"] = False
            else:
                config["Advise"]["MPC"] = True

        config["Advise"]["Thermostat_Write_Tries"] = 10


        # ==============================

        # Writes the changes the the file
        with open("./" + end_dir + "/" + f, 'wb') as o:
            yaml.dump(config, o)

