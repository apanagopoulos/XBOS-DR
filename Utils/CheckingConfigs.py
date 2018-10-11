import yaml
import glob, os
import traceback
import re

# READ ALL FILES THAT HAVE CSV EXTENSION AND CONVERT THEM TO YML EXTENSION
# COMMENT OUT IF NOT NEEDED
'''
for file in glob.glob("*.yml"):
	thisFile = file
	base = os.path.splitext(thisFile)[0]
	os.rename(thisFile, base + ".yml")
'''
# READ ALL YML FILES AND SEE IF THEY ARE PARSABLE
for file in glob.glob("*.yml"):


	#THIS PARTS CONVERTS ALL THE TABS TO SPACES INSIDE THE FILES
	#UNCOMMENT IF NEEDED
	'''
	file_contents = ''
	with open(file) as f:
		file_contents = f.read()

	file_contents = re.sub('\t', " ", file_contents)

	print('Replacing tabs in {0}'.format(file))
	with open(file, "w") as f:
		f.write(file_contents)
	'''
	try:
		with open(file, 'r') as ymlfile:
			cfg = yaml.load(ymlfile)
		print file + " WORKS"
	except Exception as exception:
		print file + " DOES NOT WORK"
		print(traceback.format_exc())  # THIS PRINTS THE STACKTRACE OF THE YAML PARSE ERROR, COMMENT IF NOT NEEDED

for file in glob.glob("*.yml"):
	thisFile = file
	base = os.path.splitext(thisFile)[0]
	os.rename(thisFile, base + ".csv")
