import sys
import glob
import os

try:
	file = sys.argv[1]
except:
	"First argument must be the config file path"
	exit()
print("Analyzing from %s"%file)

#run sorting
os.system("python3 templates_extraction.py %s"%file)
os.system("python3 sssort.py %s"%file)
os.system("python3 cluster_identification.py %s"%file)
os.system("python3 label_unknown_spikes.py %s"%file)
os.system("python3 pos_processing_amplitude.py %s"%file)
os.system("python3 pos_processing_templates.py %s"%file)
	
