import datetime
import pytz


# TODO ask someone what is going wrong with the price archiver
# TODO add energy prediction capabilities

class EnergyConsumption:
    def __init__(self, prices, interval, energy_df=None,
                 now=None, heat=0.075, cool=1.25, vent=0.02):

        self.interval = interval  # pricing time interval (mins)
        self.heat = heat  # cost of heating (kWh)
        self.cool = cool  # cost of cooling (kWh)
        self.vent = vent  # cost of ventilation (kWh)
        if now is None:
            self.now = datetime.datetime.utcnow().replace(tzinfo=pytz.timezone("UTC")).astimezone(
                         tz=pytz.timezone("America/Los_Angeles"))
        else:
            self.now = now
        self.df = energy_df
        self.prices = prices

    def calc_cost(self, action, time):
        """
        Method that calculates cost depending on time and pricing mode
        """
        if action == 'Heating' or action == '1' or action == 1:
            return (float(self.heat) * float(self.interval) / 60.) * float(self.prices[time])

        elif action == 'Cooling' or action == '2' or action == 2:
            return (float(self.cool) * float(self.interval) / 60.) * float(self.prices[time])
        elif action == 'Ventilation':
            return (float(self.vent) * float(self.interval) / 60.) * float(self.prices[time])
        elif action == 'Do Nothing' or action == '0' or action == 0:
            return 0
        else:
            print("picked wrong action %s." % str(action))
            return 0

    def get_consumption(self, action, time=None, interval=None):
        """
        Gets the energy consumption for the action at the given time.
        :param action: str or int. 
        :param time: 
        :param interval: (int) number of minutes the action is executed.
        :return: (float) consumption for the interval.
        """
        if interval is None:
            interval = self.interval
        if action == 'Heating' or action == '1' or action == 1:
            return float(self.heat) * float(interval) / 60.

        elif action == 'Cooling' or action == '2' or action == 2:
            return float(self.cool) * float(interval) / 60.
        elif action == 'Ventilation':
            return float(self.vent) * float(interval) / 60.
        elif action == 'Do Nothing' or action == '0' or action == 0:
            return 0
        else:
            print("picked wrong action %s." % str(action))
            return 0

if __name__ == '__main__':
    en_cons = EnergyConsumption([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                15)
    print en_cons.calc_cost("0", 0)
