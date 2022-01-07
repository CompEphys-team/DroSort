import sys
import glob
import os

d = sys.argv[1]

smr_files = glob.glob(d+"/*.smr")

print("Files to analize:")
print(smr_files)

for file in smr_files:
	dill_file = file[:-3]+"dill"
	file_name = file[file.rfind('/')+1:-4]

	print("Analyzing file_name")
	#Generates dill file if not exists
	if not os.path.isfile(dill_file):
		print("Generating dill file for %s"%file_name)
		os.system("python3 smr2dill.py %s %s"%(file,dill_file))

	#create new config file
	os.system("cp %s/autom_model.ini %s/%s.ini "%(d,d,file_name))
	
	#adapt config file
	adapted_dill_file = file_name.replace('/','\/')+'.dill'
	os.system("sed -i 's/data_path = file_name/data_path = %s/' %s/%s.ini"%(adapted_dill_file,d,file_name))

	os.system("sed -i 's/experiment_name = exp_name/experiment_name = %s/' %s/%s.ini"%(file_name,d,file_name))

	#run sorting
	os.system("python3 templates_extraction.py %s/model.ini"%d)
	# os.system("python3 sssort.py %s/model.ini"%d)

	# os.system("python3 cluster_identification.py %s/%s/results/"%(d,file_name))
	
	# os.system("rm %s/model.ini"%d)

