import sys
import glob
import os

try:
	file = sys.argv[1]
except:
	print("Error: First argument must be the config file path")
	exit()

print("Analyzing from %s"%file)

#get spikes events and save templates
cmd = "python templates_extraction.py %s"%file
print(cmd)
os.system(cmd)

# sorting by clustering based on detected spikes
cmd = "python sssort.py %s"%file
print(cmd)
os.system(cmd)

# cluster identification based on templates
cmd = "python cluster_identification.py %s"%file
print(cmd)
os.system(cmd)

# label unknown spikes (if found)
cmd = "python label_unknown_spikes.py %s"%file
print(cmd)
os.system(cmd)

# Run post-processing based on amplitude
cmd = "python post_processing_amplitude.py %s"%file
print(cmd)
os.system(cmd)

# Run post-processing based on templates
cmd ="python post_processing_templates.py %s"%file
print(cmd)
os.system(cmd)
	
