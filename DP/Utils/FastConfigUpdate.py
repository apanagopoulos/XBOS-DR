import os
import yaml

# Go to folder where all config files are located
config_folder_location = "../Server/Buildings/"
all_dir = os.walk(config_folder_location).next()
for d in all_dir[1]:
    print("========================")
    end_dir = config_folder_location + d + "/ZoneConfigs"
    if not os.path.isdir(end_dir):
        print("%s has not ZoneConfigs folder. Moving on to next folder" % (config_folder_location + d))
        continue

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

        # ==============================
        # Change variables in config as needed
        # print(config["Advise"]["DR_Lambda"])
        # print(config["Advise"]["Lambda"])
        config["Advise"]["DR_Lambda"] = 0.7
        config["Advise"]["General_Lambda"] = 0.7
        # ==============================

        # Writes the changes the the file
        with open("./" + end_dir + "/" + f, 'wb') as o:
            yaml.dump(config, o)

