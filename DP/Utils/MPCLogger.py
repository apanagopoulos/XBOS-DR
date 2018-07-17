import datetime
import os
import sys

sys.path.append("../Server")
import utils

LOGGER_PATH = os.path.dirname(__file__)  # this is true for now
SERVER_PATH = os.path.abspath(os.path.join(LOGGER_PATH, os.pardir)) + "/Server"  # Get parent and go to server.
NUM_NEW_LINES = 1  # number of new lines between messages

print("log path", LOGGER_PATH)
print("Server path", SERVER_PATH)


def read_line(line):
    """
    lines have format ('%Y-%m-%d %H:%M:%S UTC';Message-Side Message) This should be a placeholder line. Except when
        # the last line is system off. main and side message, e.g If we have MPC we write MPC started-lambda=0.9
    :return main_msg, side_msg, msg_time
    """
    # Get rid of any new line characters.
    line = line.replace('\n', '')

    message = line.split(" , ")
    msg_time, msg_contents = message[0], message[1:]
    line_time = utils.get_mdal_string_to_datetime(msg_time)

    # Get the main and side message, e.g If we have MPC we write MPC started-lambda=0.9
    # unpack message contents
    if len(msg_contents) == 2:
        main_msg = msg_contents[0]
        side_msg = msg_contents[1]
    else:
        main_msg = msg_contents[0]
        side_msg = None
    return main_msg, side_msg, line_time


def create_line(curr_time, main_msg, side_msg=None):
    msg = utils.get_mdal_datetime_to_string(curr_time) + " , " + main_msg
    if side_msg is not None:
        msg += " , " + side_msg
    return msg


def mpc_log(building, zone, current_time, interval, is_mpc, is_schedule, mpc_lambda, expansion, shut_down_system,
            system_shut_down_msg="Manual Setpoint Change"):
    """Logs mpc start and end times and lambda.
    :param building: string
    :param zone: string
    :param time: datetime in UTC
    :param interval: minutes"""
    # TODO get interval in more than minutes maybe? Maybe timedelta object
    # TODO maybe more than interval? What if shortest path takes too long to run
    assert not (is_mpc and is_schedule)

    path = SERVER_PATH + "/Buildings/" + building + "/" + "mpc_" + zone + ".log"

    if not os.path.exists(path):
        # create new log file
        open(path, "w+")

    messages_to_write = []

    # opening the file
    log_file = open(path, 'r')
    all_lines = log_file.readlines()

    # check if the files is empty
    if not all_lines:
        file_is_empty = True
        last_line_main_msg, last_line_side_msg, last_line_time = None, None, None
    else:
        file_is_empty = False
        # lines have format ('%Y-%m-%d %H:%M:%S UTC';Message) This should be a placeholder line. Except when
        # the last line is system off.
        last_line_main_msg, last_line_side_msg, last_line_time = read_line(all_lines[-(NUM_NEW_LINES + 1)])

    # if we have a placeholder but actually the system had ended.
    if last_line_main_msg == "Placeholder":
        # deleting the last line since we will write the place holder later or have a new message in its place
        all_lines = all_lines[:-(NUM_NEW_LINES + 1)]
        if last_line_time < current_time - datetime.timedelta(minutes=interval):
            # Replacing last line. keeping the last time and saying that then the system turned off because
            # it's the last time that we know that the system was running.
            messages_to_write.append(create_line(last_line_time, "System Off", "System did not shut off properly"))
            # setting last line to System Off because this should have been the last line.
            last_line_main_msg = "System Off"
        else:
            # Getting the previous line since that is what matters.
            last_line_main_msg, last_line_side_msg, last_line_time = read_line(all_lines[-(NUM_NEW_LINES + 1)])

    if shut_down_system:
        # shutting down system and logging it.
        messages_to_write.append(create_line(current_time, "System Off", system_shut_down_msg))
    else:
        # Handle a running system.

        # TODO why does more than an interval have to pass. EDIT: GOT rid of it
        # If the file is empty or the system had ended earlier but we are starting again.
        if file_is_empty or (last_line_main_msg == "System Off"):
            messages_to_write.append(create_line(current_time, "System On"))
            # We want to make system on our last message
            last_line_main_msg, last_line_side_msg, last_line_time = read_line(messages_to_write[-1])

        else:
            # TODO, what is this? should we use elif instead of if?
            print "WARNING: this shouldn't be happening"

        # If we are running the MPC.
        # NOTE: Just caring if we haven't already done an MPC start.
        # TODO WHY SHOULD LAST LINE NOT BE EXPANSION STARTED
        if is_mpc and not last_line_main_msg == "MPC started":
            messages_to_write.append(create_line(current_time, "MPC started", "lambda=%f" % mpc_lambda))
        else:
            # TODO, what is this? should we use elif instead of if?
            print "WARNING: this shouldn't be happening"

        # If we are running the normal schedule.
        # NOTE: Just caring if we haven't already done a Schedule start.
        # TODO WHY SHOULD LAST LINE NOT BE MPC STARTED
        if is_schedule and not last_line_main_msg == "Expansion started":
            messages_to_write.append(create_line(current_time, "Expansion started",
                                                 "Expansion=%f" % expansion))  # TODO What is expansion. percent or constant number. assuming normal schedule stuff.??
        else:
            # TODO, what is this? should we use elif instead of if?
            print "WARNING: this shouldn't be happening"

        # If the lambda changed
        if is_mpc and (last_line_main_msg == "MPC started" or last_line_main_msg == "MPC updated"):
            # compare config files lambda with last log lambda, if different
            last_lambda = last_line_side_msg.split("=")[-1]
            if float(mpc_lambda) != float(last_lambda):
                messages_to_write.append(create_line(current_time, "MPC updated", "lambda=%f" % mpc_lambda))

        # if the expansion changed.
        if is_schedule and (last_line_main_msg == "Expansion started" or last_line_main_msg == "Expansion updated"):
            # compare config files lambda with last log expansion, if different
            last_expansion = last_line_side_msg.split("=")[-1]
            if float(expansion) != float(last_expansion):
                messages_to_write.append(create_line(current_time, "Expansion updated", "Expansion=%f" % expansion))

        # adding place holder if system is running.
        messages_to_write.append(create_line(current_time, "Placeholder"))

    # adding to all_lines and adding new lines between the messages we want to write.
    all_lines += [msg + "\n" + NUM_NEW_LINES * "\n" for msg in messages_to_write]

    # writing to log file
    with open(path, "w") as log_file_write:
        log_file_write.writelines(all_lines)


if __name__ == "__main__":
    BUILDING = "ciee"
    ZONE = "HVAC_Zone_Eastzone"

    import time

    mpc_log(BUILDING, ZONE, utils.get_utc_now(), 2, True, False, 0.95, 0, False)

    mpc_log(BUILDING, ZONE, utils.get_utc_now(), 2, True, False, 0.70, 0, False)

    mpc_log(BUILDING, ZONE, utils.get_utc_now(), 2, False, True, 0.70, 10, False)

    mpc_log(BUILDING, ZONE, utils.get_utc_now(), 2, False, True, 0.70, 20, True)

    mpc_log(BUILDING, ZONE, utils.get_utc_now(), 2, False, True, 0.70, 10, False)


    time.sleep(30)
    mpc_log(BUILDING, ZONE, utils.get_utc_now(), 0.25, True, False, 0.70, 0, False)
