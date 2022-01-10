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
# from superpos_functions import *
 

#Load file
mpl.rcParams['figure.dpi'] = 300
fig_format = '.png'


# get config
config_path = Path(os.path.abspath(sys.argv[1]))
Config = configparser.ConfigParser()
Config.read(config_path)
print_msg('config file read from %s' % config_path)

# handling paths and creating output directory
data_path = Path(Config.get('path','data_path'))
if not data_path.is_absolute():
    data_path = config_path.parent / data_path

exp_name = Config.get('path','experiment_name')
results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots' / 'post_processing'

os.makedirs(plots_folder, exist_ok=True)

Blk=get_data(results_folder/"result.dill")
SpikeInfo = pd.read_csv(results_folder/"SpikeInfo.csv")

unit_column = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
SpikeInfo = SpikeInfo.astype({unit_column: str})
units = get_units(SpikeInfo,unit_column)

#Load Templates
Waveforms= np.load(results_folder/"Templates_ini.npy")
n_samples = Waveforms[:,0].size

new_column = 'unit_labeled'

if len(units) != 3:
	print("Three units needed, only %d found in SpikeInfo"%len(units))
	exit()

if '-2' in units or new_column in SpikeInfo.keys():
    print_msg("Clusters already assigned")
    print(SpikeInfo[unit_column].value_counts())
    exit()


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
print_msg("\t\t%s" % str(units))
print_msg("\t\t%s" % str(distances_a))
print_msg("Distances to b: ")
print_msg("\t\t%s" % str(units))
print_msg("\t\t%s" % str(distances_b))

# Get best assignations
a_unit = units[np.argmin(distances_a)]
b_unit = units[np.argmin(distances_b)]
non_unit = [unit for unit in units if a_unit not in unit and b_unit not in unit][0]


asigs = {a_unit: 'A', b_unit: 'B', non_unit: '?'}
print_msg("Final assignation: %s" % asigs)

# plot assignation
outpath = plots_folder / ("cluster_reassignation" + fig_format)
plot_means(means, units, template_a, template_b, asigs=asigs, outpath=outpath)

# create new column with reassigned labels
SpikeInfo[new_column] = copy.deepcopy(SpikeInfo[unit_column].values)
non_unit_rows = SpikeInfo.groupby(new_column).get_group(non_unit)
a_unit_rows = SpikeInfo.groupby(new_column).get_group(a_unit)
b_unit_rows = SpikeInfo.groupby(new_column).get_group(b_unit)

SpikeInfo.loc[non_unit_rows.index, new_column] = '-2'
SpikeInfo.loc[a_unit_rows.index, new_column] = 'a'
SpikeInfo.loc[b_unit_rows.index, new_column] = 'b'

units = get_units(SpikeInfo, new_column)
Blk = populate_block(Blk, SpikeInfo, new_column, units)

# store SpikeInfo
outpath = results_folder / 'SpikeInfo.csv'
print_msg("saving SpikeInfo to %s" % outpath)
SpikeInfo.to_csv(outpath)
