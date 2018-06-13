import os
import yaml

all_dir = os.walk("./").next()
for d in all_dir[1]:
    print("========================")
    end_dir = d + "/ZoneConfigs"
    if not os.path.isdir(end_dir):
        print("%s has not ZoneConfigs folder. Moving on to next folder" % d)
        continue

    files = os.walk(end_dir).next()[2]

    print("in dir: ", end_dir)
    for f in files:
        # Loads the configs
        with open("./" + end_dir + "/" + f, 'r') as o:
            config = yaml.load(o)

        # Change variables in config as needed
        print(config["Advise"]["DR_Lambda"])
        print(config["Advise"]["Lambda"])
        #             config["Advise"]["DR_Lambda"] = 0.7
        #             config["Advise"]["Lambda"] = 0.7

        # Writes the changes the the file
        with open("./" + end_dir + "/" + f, 'wb') as o:
            yaml.dump(config, o)

