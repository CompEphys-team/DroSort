import matplotlib.pyplot as plt
import pandas as pd
from sys import path
import configparser

# from superpos_functions import *
from sssio import * 
from plotters import *
from functions import *
from posprocessing_functions import *

# path = sys.argv[1]

################################################################
##
## Posprocessing reasigning clusters by neighbours amplitude.
##
################################################################

################################################################
##  
##              Load clustering result
##
################################################################


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
plots_folder = results_folder / 'plots' / 'pos_processing' / 'amplitude'

os.makedirs(plots_folder, exist_ok=True)

Blk=get_data(results_folder/"result.dill")
SpikeInfo = pd.read_csv(results_folder/"SpikeInfo.csv")

unit_column = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
SpikeInfo = SpikeInfo.astype({unit_column: str})
units = get_units(SpikeInfo,unit_column)
Templates= np.load(results_folder/"Templates_ini.npy")


fig_format='.png'
mpl.rcParams['figure.dpi'] = 300


print_msg("Number of spikes in trace: %d"%SpikeInfo[unit_column].size)
# print_msg("Number of good spikes: %d"%len(SpikeInfo.groupby(['good']).get_group(True)[unit_column]))
# print_msg("Number of bad spikes: %d"%len(SpikeInfo.groupby(['good']).get_group(False)[unit_column]))
print_msg("Number of clusters: %d"%len(units))
print_msg("SpikeInfo units:")
print(SpikeInfo[unit_column].value_counts())

new_column = 'unit_amplitude'

################################################################
##  
##              Reassing by neighbor's amplitude
##  
################################################################

spike_ids = SpikeInfo['id'].values  
#third cluster fix
if '-2' in units:
    units.remove('-2') 
if len(units) > 2:
    print_msg("Too many units: %s. Try run cluster_identification.py first"%units)
    exit()
if new_column in SpikeInfo.keys():
    print_msg("Posprocessing by amplitude is already in SpikeInfo")
    exit()

dict_units = {u:i for i,u in enumerate(units)}

st = Blk.segments[0].spiketrains[0]
Seg = Blk.segments[0]

n_samples = Templates[:,0].size 

new_labels = copy.deepcopy(SpikeInfo[unit_column].values)

# Calculate neighbors time as spike time * n_spikes + isi mean time
n_neighbors = 6#TODO add to config file
neighbors_t = get_neighbors_time(Seg.analogsignals[0],st,n_samples,n_neighbors)
print_msg("Surrounding time for neighbors: %f"%neighbors_t)
#Note: neighbors_t is a fixed time, when checking n neighbours ignoring time, 
#       the distance between spikes could be too high.


#new column in SpikeInfo with changes
SpikeInfo[new_column] = new_labels
ids = []

print_msg("Processing comparation")
for i, (spike_id,org_label) in enumerate(zip(spike_ids,SpikeInfo[unit_column])):
    # if org_label == '-1':
    #     continue
    if int(org_label) < 0:
        continue

    spike = Templates[:, spike_id].T
    # ampl = max(spike)-min(spike)
    ampl = max(spike)-min(spike[spike.size//2:]) #half spike to avoid noise

    new_label = units[(dict_units[org_label]+1)%2]

    #Get amplitude from neighbors in spike_t +/- neighbors_t. 
    sur_ampl = get_neighbors_amplitude(st,Templates,SpikeInfo,unit_column,org_label,idx=spike_id,t=neighbors_t)
    sur_ampl_new = get_neighbors_amplitude(st,Templates,SpikeInfo,unit_column,new_label,idx=spike_id,t=neighbors_t)

    if abs(ampl-sur_ampl) > abs(ampl-sur_ampl_new):
        SpikeInfo[new_column][i] = new_label
        new_labels[i] = new_label
        ids.append(i)

        #plot change
        zoom = [st.times[spike_id]-neighbors_t*pq.s,st.times[spike_id]+neighbors_t*pq.s]
        outpath = plots_folder / ("cluster_changed_amplitude_%d"%i+fig_format)
        fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, [unit_column, new_column], zoom=zoom, save=outpath,wsize=n_samples)


print_msg("Num of final changes %d"%np.sum(~(SpikeInfo[unit_column]==new_labels).values))

# store SpikeInfo
outpath = results_folder / 'SpikeInfo.csv'
print_msg("saving SpikeInfo to %s" % outpath)
SpikeInfo.to_csv(outpath)

# repopulate block?
# print(ids)
# Blk = populate_block(Blk,SpikeInfo,new_column,units)
# st = Blk.segments[0].spiketrains[0]




print_msg("Saving SpikeInfo, Blk and Spikes into disk")
print_msg("Current units",units)
save_all(results_folder,Config,SpikeInfo,Blk,units)

print_msg("Done")