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
##
################################################################

################################################################
##  
##              Load clustering result
##
################################################################

Blk=get_data(sys.argv[1]+"/result.dill")

SpikeInfo = pd.read_csv(sys.argv[1]+"/SpikeInfo.csv")

print(SpikeInfo.keys())

last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
unit_column = last_unit_col
SpikeInfo = SpikeInfo.astype({last_unit_col: str})
units = get_units(SpikeInfo,unit_column)

Templates= np.load(sys.argv[1]+"/Templates.npy")



print_msg("Number of spikes in trace: %d"%SpikeInfo[unit_column].size)
print_msg("Number of clusters: %d"%len(units))

################################################################
##  
##              Reassing by neighbor's amplitude
##  
################################################################

spike_ids = SpikeInfo['id'].values    
dict_units = {u:i for i,u in enumerate(units)}

st = Blk.segments[0].spiketrains[0]
Seg = Blk.segments[0]

n_samples = Templates[:,0].size

new_labels = copy.deepcopy(SpikeInfo[unit_column].values)
n_neighbors = 3 #TODO add to config file

#Time of one spike * number of spikes * 2 (approximate isis)
neighbors_t = (n_samples/1000)*n_neighbors 
print(neighbors_t)
#new column in SpikeInfo with changes
SpikeInfo['unit_pospro'] = new_labels
ids = []

for i, (spike_id,org_label) in enumerate(zip(spike_ids,SpikeInfo[unit_column])):
    if org_label == '-1':
        continue

    spike = Templates[:, spike_id].T
    # ampl = max(spike)-min(spike)
    ampl = max(spike)-min(spike[spike.size//2:]) #half spike to avoid noise

    new_label = units[(dict_units[org_label]+1)%2]

    sur_ampl = get_neighbors_amplitude(st,Templates,SpikeInfo,unit_column,org_label,idx=spike_id,t=neighbors_t)
    sur_ampl_new = get_neighbors_amplitude(st,Templates,SpikeInfo,unit_column,new_label,idx=spike_id,t=neighbors_t)

    dur = get_duration(spike)
    sur_dur = get_neighbors_amplitude(st,Templates,SpikeInfo,unit_column,org_label,idx=spike_id,t=neighbors_t)
    sur_dur_new = get_neighbors_amplitude(st,Templates,SpikeInfo,unit_column,new_label,idx=spike_id,t=neighbors_t)

    # if abs(ampl-sur_ampl) > abs(ampl-sur_ampl_new)+0.05 and abs(dur-sur_dur) > abs(dur-sur_dur_new):
    if abs(ampl-sur_ampl) > abs(ampl-sur_ampl_new):
        SpikeInfo['unit_pospro'][i] = new_label
        new_labels[i] = new_label
        ids.append(i)

        # if st.times[spike_id] > 6.2 and st.times[spike_id] < 6.5:
        # zoom = [st.times[spike_id]-neighbors_t*pq.s,st.times[spike_id]+neighbors_t*pq.s]
        # print(ampl,sur_ampl,sur_ampl_new,st.times[spike_id])
        # print(ampl,abs(ampl-sur_ampl),abs(ampl-sur_ampl_new),SpikeInfo[unit_column][i],SpikeInfo['unit_pospro'][i])
        # print(dur,sur_dur,sur_dur_new,st.times[spike_id])
        # print(dur,abs(dur-sur_dur),abs(dur-sur_ampl_new),SpikeInfo[unit_column][i],SpikeInfo['unit_pospro'][i])
        # fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, unit_column, 'unit_pospro', zoom=zoom, save=None,wsize=n_samples)

        # plt.show()  

print_msg("Num of final changes %d"%np.sum(~(SpikeInfo[unit_column]==new_labels).values))


print(ids)
Blk = populate_block(Blk,SpikeInfo,'unit_pospro',units)
st = Blk.segments[0].spiketrains[0]

for i, (spike_id,org_label) in enumerate(zip(spike_ids,SpikeInfo['unit_pospro'])):
    if i in ids:
        zoom = [st.times[spike_id]-neighbors_t*pq.s,st.times[spike_id]+neighbors_t*pq.s]
        print(ampl,sur_ampl,sur_ampl_new,st.times[spike_id])
        fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, unit_column, 'unit_pospro', zoom=zoom, save=None,wsize=n_samples)
        axes[0].plot(st.times[spike_id],1,'.')
        axes[1].plot(st.times[spike_id],1,'.')
        plt.show()



################################################################
##  
##              Unassign by shapes
##  
################################################################


# Get average shapes

