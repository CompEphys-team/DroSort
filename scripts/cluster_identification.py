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
import matplotlib.pyplot as plt 

#Load file
mpl.rcParams['figure.dpi'] = 300
fig_format = '.png'


# get config
config_path = Path(os.path.abspath(sys.argv[1]))
sssort_path = os.path.dirname(os.path.abspath(sys.argv[0]))
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
fs = Blk.segments[0].analogsignals[0].sampling_rate
n_samples= np.array(Config.get('spike model','template_window').split(','),dtype='float32')/1000.0
n_samples= np.array(n_samples*fs, dtype= int)

new_column = 'unit_labeled'

#if len(units) != 3:
#	print("Three units needed, only %d found in SpikeInfo"%len(units))
#	exit()

if new_column in SpikeInfo.keys():
    print_msg("Clusters already assigned")
    print(SpikeInfo[unit_column].value_counts())
    exit()


#Load model templates 
template_a = np.load(os.path.join(sssort_path,"templates/template_a.npy"))
template_b = np.load(os.path.join(sssort_path,"templates/template_b.npy"))

# templates and waveforms need to be put on comparable shape and size
tmid_a= np.argmax(template_a)
tmid_b= np.argmax(template_b)
left= np.amin([ tmid_a, tmid_b, n_samples[0] ])
right= np.amin([ len(template_a)-tmid_a, len(template_b)-tmid_b, n_samples[1] ])

template_a = template_a[tmid_a-left:tmid_a+right]
template_b = template_b[tmid_b-left:tmid_b+right]
Waveforms= Waveforms[n_samples[0]-left:n_samples[0]+right,:]

print_msg("Current units: %s"%units)

distances_a=[]
distances_b=[]
means=[]

mode='peak'

print_msg("Computing best assignment")
#Compare units to templates
mean_waveforms= {}
amplitude= []
for unit in units:
    unit_ids = SpikeInfo.groupby(unit_column).get_group(unit)['id']
    waveforms = Waveforms[:,unit_ids]

    waveforms = np.array([np.array(align_to(t,mode)) for t in waveforms.T])

    mean_waveforms[unit] = np.average(waveforms,axis=0)
    amplitude.append(np.max(mean_waveforms[unit])-np.min(mean_waveforms[unit]))

max_ampl= np.max(amplitude)
norm_factor= (np.max(template_a)-np.min(template_a))/max_ampl

for unit in units:
    #plt.figure()
    #plt.plot(mean_waveforms[unit]*norm_factor)
    #plt.plot(template_a)
    #plt.plot(template_b)
    #plt.show()
    d_a = np.linalg.norm(mean_waveforms[unit]*norm_factor-template_a)
    d_b = np.linalg.norm(mean_waveforms[unit]*norm_factor-template_b)
    #d_a = metrics.pairwise.euclidean_distances(mean_waveforms.reshape(1,-1),template_a.reshape(1,-1)).reshape(-1)[0]
    #d_b = metrics.pairwise.euclidean_distances(mean_waveforms.reshape(1,-1),template_b.reshape(1,-1)).reshape(-1)[0]
    
    #compute distances by mean of distances
    # distances_a = metrics.pairwise.euclidean_distances(waveforms,template_a.reshape(1,-1)).reshape(-1)
    # distances_b = metrics.pairwise.euclidean_distances(waveforms,template_b.reshape(1,-1)).reshape(-1)

    distances_a.append(d_a)
    distances_b.append(d_b)
    means.append(mean_waveforms[unit])


print_msg("Distances to a: ")
print_msg("\t\t%s" % str(units))
print_msg("\t\t%s" % str(distances_a))
print_msg("Distances to b: ")
print_msg("\t\t%s" % str(units))
print_msg("\t\t%s" % str(distances_b))

# Get best assignments
a_unit = units[np.argmin(distances_a)]
b_unit = units[np.argmin(distances_b)]
if len(units) > 2:
    non_unit = [unit for unit in units if a_unit not in unit and b_unit not in unit][0]


asigs = {a_unit: 'A', b_unit: 'B'}
if len(units) > 2:
    asigs[non_unit]= '?'
print_msg("Final assignation: %s" % asigs)

# plot assignments
outpath = plots_folder / ("cluster_reassignments" + fig_format)
plot_means(means, units, template_a, template_b, asigs=asigs, outpath=outpath)

# create new column with reassigned labels
SpikeInfo[new_column] = copy.deepcopy(SpikeInfo[unit_column].values)
if len(units) > 2:
    non_unit_rows = SpikeInfo.groupby(new_column).get_group(non_unit)
    SpikeInfo.loc[non_unit_rows.index, new_column] = '-2'

a_unit_rows = SpikeInfo.groupby(new_column).get_group(a_unit)
SpikeInfo.loc[a_unit_rows.index, new_column] = 'a'
b_unit_rows = SpikeInfo.groupby(new_column).get_group(b_unit)
SpikeInfo.loc[b_unit_rows.index, new_column] = 'b'

units = get_units(SpikeInfo, new_column)
Blk = populate_block(Blk, SpikeInfo, new_column, units)

# store SpikeInfo
outpath = results_folder / 'SpikeInfo.csv'
print_msg("saving SpikeInfo to %s" % outpath)
SpikeInfo.to_csv(outpath)
