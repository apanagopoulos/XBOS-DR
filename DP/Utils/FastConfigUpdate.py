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

    # Get ready for DR-event so we can run on server
    building_config["Server"] = False

    # Setting DR start and end
    building_config["Pricing"]["DR_Start"] = "14:00"
    building_config["Pricing"]["DR_Finish"] = "18:00"

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
        config["Advise"]["DR_Lambda"] = 0.7
        config["Advise"]["General_Lambda"] = 0.995

        # Decide whether to run MPC.
        if "Cooling_Consumption_Stage_2" in config["Advise"] and not directory == "south-berkeley-senior-center": # we will test thermal model on a two stage cooling building.
            config["Advise"]["MPC"] = False
            print("%s zone will NOT run MPC." % f)
        else:
            config["Advise"]["MPC"] = True
            print("%s zone will run MPC." % f)

        # ==============================

        # Writes the changes the the file
        with open("./" + end_dir + "/" + f, 'wb') as o:
            yaml.dump(config, o)

