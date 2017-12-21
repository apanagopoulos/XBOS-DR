
def toBase(n,base):
	convertString = "0123456789ABCDEF"
	if n < base:
		return convertString[n]
	else:
		return toBase(n//base,base) + convertString[n%base]
 

class Safety:
	def __init__(self, max_temperature=86, min_temperature=54, noZones = 1):
		self.max_temperature = max_temperature
		self.min_temperature = min_temperature
		self.noZones = noZones

	def safety_actions(self, temps):
		nonSafe = []
		for i in range(1, self.noZones+1):
			if temps[i-1] > self.max_temperature: 
				nonSafe.append(i)
			elif temps[i-1] < self.min_temperature:
				nonSafe.append(-i)
		actions = [toBase(i,3).zfill(self.noZones) for i in range(3**self.noZones)]
		for i in nonSafe:
			if i >= 0:
				actions = list(filter(lambda x: x[i-1] == '1', actions))
			else:
				actions2 = []
				for action in actions:
					str_temp = ''
					for j in range(len(action)):
						if j != -i-1:
							str_temp += action[j]
						else:
							str_temp += '2'
					actions2.append(str_temp)
				actions = actions2
				actions = list(set(actions))
		return actions


	def safety_check(self, temp):
		flag = False
		for i in range(self.noZones):
			if temp[i] > self.max_temperature or temp[i] < self.min_temperature:
				flag = True

		return flag
