def toBase(n, base):
    """
    Utility function, converts certain number to another base
    """
    convertString = "0123456789ABCDEF"
    if n < base:
        return convertString[n]
    else:
        return toBase(n // base, base) + convertString[n % base]


class Safety:
    """
    # this is the module that generates all the possible combinations of heating/cooling/do nothing actions
    """

    def __init__(self, safety_constraints, noZones=1):
        self.safety_constraints = safety_constraints
        self.noZones = noZones

    def safety_actions(self, temps, time):
        """
        Calculate a string of all the available actions according to the safety setpoints
        Could work for N zones
        """
        nonSafe = []
        for i in range(1, self.noZones + 1):
            if temps[i - 1] > self.safety_constraints[time][1]:
                nonSafe.append(i)
            elif temps[i - 1] < self.safety_constraints[time][0]:
                nonSafe.append(-i)
        actions = [toBase(i, 3).zfill(self.noZones) for i in
                   range(3 ** self.noZones)]  # change the 3 in toBase() for more actions
        for i in nonSafe:
            if i >= 0:
                actions = list(filter(lambda x: x[i - 1] == '2', actions))
            else:
                actions2 = []
                for action in actions:
                    str_temp = ''
                    for j in range(len(action)):
                        if j != -i - 1:
                            str_temp += action[j]
                        else:
                            str_temp += '1'
                    actions2.append(str_temp)
                actions = actions2
                actions = list(set(actions))
        return actions

    def safety_check(self, temp, time):
        """
        Check if a temp is withing the safety limits
        """
        flag = False
        for i in range(self.noZones):
            if temp[i] > self.safety_constraints[time][1] or temp[i] < self.safety_constraints[time][0]:
                flag = True


        return flag
