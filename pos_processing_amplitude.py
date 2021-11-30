import matplotlib.pyplot as plt
import pandas as pd
from sssio import * 
from plotters import *
from functions import *
from sys import path
from superpos_functions import *

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

results_folder = Path(os.path.abspath(sys.argv[1]))

# results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots' / 'pos_processing'
os.makedirs(plots_folder, exist_ok=True)

fig_format='.png'

mpl.rcParams['figure.dpi'] = 300

Blk=get_data(sys.argv[1]+"/result.dill")
SpikeInfo = pd.read_csv(sys.argv[1]+"/SpikeInfo.csv")

# last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
unit_column = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
# unit_column = last_unit_col
SpikeInfo = SpikeInfo.astype({unit_column: str})
units = get_units(SpikeInfo,unit_column)

Templates= np.load(sys.argv[1]+"/Templates_ini.npy")

print_msg("Number of spikes in trace: %d"%SpikeInfo[unit_column].size)
print_msg("Number of good spikes: %d"%len(SpikeInfo.groupby(['good']).get_group(True)[unit_column]))
# print_msg("Number of bad spikes: %d"%len(SpikeInfo.groupby(['good']).get_group(False)[unit_column]))
print_msg("Number of clusters: %d"%len(units))

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

dict_units = {u:i for i,u in enumerate(units)}

st = Blk.segments[0].spiketrains[0]
Seg = Blk.segments[0]

n_samples = Templates[:,0].size 

new_labels = copy.deepcopy(SpikeInfo[unit_column].values)
n_neighbors = 3 #TODO add to config file

# #Time of one spike * number of spikes * 2 (approximate isis)
# neighbors_t = (n_samples/1000)*n_neighbors 
# print(neighbors_t)
# # TODO: fix use sampling and no 1000 ?!?!?!
# dt = Seg.analogsignals[0].times[1]-Seg.analogsignals[0].times[0]
# dt = dt.item()
# neighbors_t = (n_samples*dt)*n_neighbors 
# print(neighbors_t)

#Calculate neighbors time as n first spikes + ISIs time
neighbors_t = st.times[n_neighbors].item()
print_msg("Surrounding time for neighbors: %f"%neighbors_t)

#new column in SpikeInfo with changes
SpikeInfo['unit_amplitude'] = new_labels
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

    # dur = get_duration(spike)
    # sur_dur = get_neighbors_amplitude(st,Templates,SpikeInfo,unit_column,org_label,idx=spike_id,t=neighbors_t)
    # sur_dur_new = get_neighbors_amplitude(st,Templates,SpikeInfo,unit_column,new_label,idx=spike_id,t=neighbors_t)

    if abs(ampl-sur_ampl) > abs(ampl-sur_ampl_new):
        SpikeInfo['unit_amplitude'][i] = new_label
        new_labels[i] = new_label
        ids.append(i)

        #plot change
        zoom = [st.times[spike_id]-neighbors_t*pq.s,st.times[spike_id]+neighbors_t*pq.s]
        outpath = plots_folder / ("cluster_changed_amplitude_%d"%i+fig_format)
        fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, [unit_column, 'unit_amplitude'], zoom=zoom, save=outpath,wsize=n_samples)

        # if st.times[spike_id] > 6.2 and st.times[spike_id] < 6.5:
            # print(ampl,sur_ampl,sur_ampl_new,st.times[spike_id])
            # print(ampl,abs(ampl-sur_ampl),abs(ampl-sur_ampl_new),SpikeInfo[unit_column][i],SpikeInfo['unit_amplitude'][i])
            # print(dur,sur_dur,sur_dur_new,st.times[spike_id])
            # print(dur,abs(dur-sur_dur),abs(dur-sur_ampl_new),SpikeInfo[unit_column][i],SpikeInfo['unit_amplitude'][i])

print_msg("Num of final changes %d"%np.sum(~(SpikeInfo[unit_column]==new_labels).values))

# store SpikeInfo
outpath = results_folder / 'SpikeInfo.csv'
print_msg("saving SpikeInfo to %s" % outpath)
SpikeInfo.to_csv(outpath)

# repopulate block?
# print(ids)
# Blk = populate_block(Blk,SpikeInfo,'unit_amplitude',units)
# st = Blk.segments[0].spiketrains[0]
