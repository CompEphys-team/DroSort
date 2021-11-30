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
from sssio import *
from superpos_functions import *
 

#Load file
mpl.rcParams['figure.dpi'] = 300

results_folder = Path(os.path.abspath(sys.argv[1]))

# results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots' / 'pos_processing'
os.makedirs(plots_folder, exist_ok=True)

fig_format = '.png'

#load Blk
Blk=get_data(sys.argv[1]+"/result.dill")
Seg = Blk.segments[0]

#load SpikeInfo
SpikeInfo = pd.read_csv(sys.argv[1]+"/SpikeInfo.csv")
unit_column = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
SpikeInfo = SpikeInfo.astype({unit_column: str})
units = get_units(SpikeInfo,unit_column)

if len(units) != 3:
	print("Three units needed, only %d found in SpikeInfo"%len(units))
	exit()

if '-2' in units or 'unit_labeled' in SpikeInfo.keys():
    print_msg("Clusters already assigned")
    exit()

#Load Templates
Waveforms= np.load(sys.argv[1]+"/Templates_ini.npy")
n_samples = Waveforms[:,0].size

#Load model templates 
template_a = np.load("./templates/template_a.npy")
template_b = np.load("./templates/template_b.npy")

template_a = template_a[:Waveforms.shape[0]]
template_b = template_b[:Waveforms.shape[0]]

print_msg("Current units: %s"%units)

distances_a=[]
distances_b=[]
means=[]

mode='peak'

print_msg("Computing best assignation")
#Compare units to templates
for unit in units:
    unit_ids = SpikeInfo.groupby(unit_column).get_group(unit)['id']
    waveforms = Waveforms[:,unit_ids]

    waveforms = np.array([np.array(align_to(t,mode)) for t in waveforms.T])

    mean_waveforms = np.average(waveforms,axis=0)

    d_a = metrics.pairwise.euclidean_distances(mean_waveforms.reshape(1,-1),template_a.reshape(1,-1)).reshape(-1)[0]
    d_b = metrics.pairwise.euclidean_distances(mean_waveforms.reshape(1,-1),template_b.reshape(1,-1)).reshape(-1)[0]
    
    #compute distances by mean of distances
    # distances_a = metrics.pairwise.euclidean_distances(waveforms,template_a.reshape(1,-1)).reshape(-1)
    # distances_b = metrics.pairwise.euclidean_distances(waveforms,template_b.reshape(1,-1)).reshape(-1)

    distances_a.append(d_a)
    distances_b.append(d_b)
    means.append(mean_waveforms)


print_msg("Distances to a: ")
print_msg("\t\t%s"%str(units))
print_msg("\t\t%s"%str(distances_a))
print_msg("Distances to b: ")
print_msg("\t\t%s"%str(units))
print_msg("\t\t%s"%str(distances_b))

#Get best assignations
a_unit = units[np.argmin(distances_a)]
b_unit = units[np.argmin(distances_b)]
non_unit = [unit for unit in units if a_unit not in unit and b_unit not in unit][0]


asigs = {a_unit:'A',b_unit:'B',non_unit:'?'}
print_msg("Final assignation: %s"%asigs)

#plot assignation
outpath = plots_folder / ("cluster_reassignation" + fig_format)
plot_means(means,units,template_a,template_b,asigs=asigs,outpath=outpath)

# create new column with reassigned labels
SpikeInfo['unit_labeled'] = copy.deepcopy(SpikeInfo[unit_column].values)
Df = SpikeInfo.groupby('unit_labeled').get_group(non_unit)
SpikeInfo.loc[Df.index, 'unit_labeled'] = '-2'

# store SpikeInfo
outpath = results_folder / 'SpikeInfo.csv'
print_msg("saving SpikeInfo to %s" % outpath)
SpikeInfo.to_csv(outpath)
