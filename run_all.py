import sys
import glob
import os

try:
	file = sys.argv[1]
except:
	"First argument must be the config file path"

	#run sorting
	os.system("python3 templates_extraction.py %s"%file)
	os.system("python3 sssort.py %s/model.ini"%d)
	os.system("python3 cluster_identification.py %s/model.ini"%d)
	os.system("python3 pos_processing_amplitude.py %s/model.ini"%d)
	os.system("python3 pos_processing_templates.py %s/model.ini"%d)
	
	os.system("rm %s/model.ini"%d)

