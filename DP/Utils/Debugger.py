import os


def debug_print(now, building, zone, adv, safety_constraints, prices, building_setpoints, time, file=True):
    if not file:
        print now.strftime('%Y-%m-%d %H:%M:%S')
        print "MPC Actions:"
        print "Old"
        action_list = []
        counter = 0
        while True:
            try:
                action_list.append(adv.advise_unit.g[adv.path[counter]][adv.path[counter + 1]]['action'])
                counter += 1
            except:
                break
        print action_list
        print "New"
        action_list = []
        counter = 0
        while True:
            try:
                action_list.append(adv.advise_unit.g.node[adv.path[counter]]['best_action'])
                counter += 1
            except:
                break

        print action_list[:-1]
        print "Temperatures following the MPC path:"
        print [i.temps[0] for i in adv.path]
        print "Safety Constraints:"
        print safety_constraints
        print "Prices:"
        print prices
        print "Occ Predictions:"
        print adv.occ_predictions
        print "Setpoints:"
        print building_setpoints
        print "Time needed for the shortest path:"
        print time
    else:

        write_string = "\n\n" + now.strftime('%Y-%m-%d %H:%M:%S') + "\n"
        write_string += "MPC Actions: "
        action_list = []
        counter = 0
        while True:
            try:
                action_list.append(adv.advise_unit.g.node[adv.path[counter]]['best_action'])
                counter += 1
            except:
                break

        write_string += str(action_list[:-1])
        write_string += "\nTemperatures following the MPC path:\n"
        write_string += str([i.temps[0] for i in adv.path])
        write_string += "\nSafety Constraints:\n"
        write_string += str(safety_constraints)
        write_string += "\nPrices:\n"
        write_string += str(prices)
        write_string += "\nOcc Predictions:\n"
        write_string += str(adv.occ_predictions)
        write_string += "\nSetpoints:\n"
        write_string += str(building_setpoints)
        write_string += "\nTime needed to calculate the shortest path:\n"
        write_string += str(time) + "\n"

        if not os.path.exists("../Server/Buildings/" + building + "/Logs"):
            os.makedirs("../Server/Buildings/" + building + "/Logs")

        if os.path.exists("../Server/Buildings/" + building + "/Logs/" + zone + ".log"):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not
        logfile = open("../Server/Buildings/" + building + "/Logs/" + zone + ".log", append_write)
        logfile.write(write_string)
        logfile.close()
