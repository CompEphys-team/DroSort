########################################################################################################
# 
#  Loads neurons template and labels given clusters based on similarity to the templates
# 
########################################################################################################

# sys
import sys
import os
import copy
import dill
import shutil
import configparser
from pathlib import Path
from tqdm import tqdm

# sci
import scipy as sp
import numpy as np

# ephys
import neo
import elephant as ele

# own
from functions import *
from plotters import *
import sssio

 

#Load file

results_folder = Path(os.path.abspath(sys.argv[1]))

print(results_folder)
# results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots'
print(plots_folder)
fig_format = '.png'

# Blk=get_data(sys.argv[1]+"/result.dill")

SpikeInfo = pd.read_csv(sys.argv[1]+"/SpikeInfo.csv")

print(SpikeInfo.keys())

last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
unit_column = last_unit_col
SpikeInfo = SpikeInfo.astype({last_unit_col: str})
units = get_units(SpikeInfo,unit_column)

if len(units) != 3:
	print("Three units needed, only %d found in SpikeInfo"%len(units))
	exit()



Waveforms= np.load(sys.argv[1]+"/Templates_ini.npy")

#Load templates 
template_a = np.load("./templates/template_a.npy")
template_b = np.load("./templates/template_b.npy")

template_a = template_a[:Waveforms.shape[0]]
template_b = template_b[:Waveforms.shape[0]]

#Show loaded templates
plt.subplot(1,2,1)
plt.plot(template_a)
plt.title("A")
plt.subplot(1,2,2)
plt.plot(template_b)
plt.title("B")
plt.show()

print(units)

distances_a=[]
distances_b=[]
#Compare units to templates
for unit in units:
	unit_ids = SpikeInfo.groupby(unit_column).get_group(unit)['id']
	waveforms = Waveforms[:,unit_ids]

	plt.plot(waveforms,color='b')
	plt.title(unit)
	plt.ylim(-1,1)
	plt.show()

	mean_waveforms = np.average(waveforms.T,axis=0)

	distances_a.append(metrics.pairwise.euclidean_distances(mean_waveforms.reshape(1,-1),template_a.reshape(1,-1)).reshape(-1))
	distances_b.append(metrics.pairwise.euclidean_distances(mean_waveforms.reshape(1,-1),template_b.reshape(1,-1)).reshape(-1))

a_unit = units[np.argmin(distances_a)]
b_unit = units[np.argmin(distances_b)]

non_unit = [x for x in units if a_unit not in x and b_unit not in x][0]

print(distances_a,distances_b)
print("A B ?")
print(a_unit,b_unit,non_unit)


SpikeInfo['unit_labeled'] = copy.deepcopy(SpikeInfo[unit_column].values)


Df = SpikeInfo.groupby('unit_labeled').get_group(non_unit)
SpikeInfo.loc[Df.index, 'unit_labeled'] = '-2'

#TODO: save SpikeInfo