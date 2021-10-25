import matplotlib.pyplot as plt
import pandas as pd
from sssio import * 
from plotters import *
from functions import *

# path = sys.argv[1]

Blk=get_data(sys.argv[1]+"/result.dill")


SpikeInfo = pd.read_csv(sys.argv[1]+"/SpikeInfo.csv")

print(SpikeInfo.keys())
print(SpikeInfo.describe())

last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]

units = get_units(SpikeInfo,last_unit_col)
print(units)

colors = get_colors(units)

# for seg in Blk.segments:
#     print(seg)
#     for n,asig in enumerate(seg.analogsignals):
#         plt.subplot(len(seg.analogsignals),1,n+1)
#         plt.plot(asig.times,asig.magnitude)

#         if n==0:
#             for i,sp in enumerate(seg.spiketrains[0]):
#                 unit =SpikeInfo[SpikeInfo['id']==i][last_unit_col].values[0] 
#                 col = colors[str(unit)]

#                 plt.plot(seg.spiketrains[0].times[i],seg.spiketrains[0].waveforms.reshape(seg.spiketrains[0].times.size)[i],'.',color=col)

# plt.show()


Templates= np.load(sys.argv[1]+"/Templates.npy")
unit_column = last_unit_col

units = get_units(SpikeInfo,unit_column)

# amplitudes = get_units_amplitudes(Templates,SpikeInfo,unit_column,lim=10)

spike_ids = SpikeInfo['id'].values    
dict_units = {u:i for i,u in enumerate(units)}
print(dict_units)


st = Blk.segments[0].spiketrains[0]
print(st.size)
Seg = Blk.segments[0]
n_samples = Templates[:,0].size
print(n_samples)
new_labels = copy.deepcopy(SpikeInfo[unit_column].values)

for i, (spike_id,org_label) in enumerate(zip(spike_ids,SpikeInfo[unit_column])):
    
    org_label = str(org_label)
    if org_label == '-1':
        continue
    spike = Templates[:, spike_id].T
    ampl = max(spike)-min(spike)

    new_label = units[(dict_units[org_label]+1)%2]

    zoom = [st.times[spike_id]-0.3*pq.s,st.times[spike_id]+0.3*pq.s]

    fig, axes=plot_fitted_spikes(Seg, 0, Templates, SpikeInfo, last_unit_col, zoom=zoom, save=None,wsize=n_samples)
    axes[1].plot(st.times[spike_id],spike[spike.size//2],'.',markersize=10,color='r')

    #TODO: fix and analyze neighbours by idx not cluster ?
    sur_ampl = get_neighbours_amplitude(st,Templates,SpikeInfo,unit_column,org_label,idx=spike_id,n=3,ax=axes,id_=2)
    sur_ampl_new = get_neighbours_amplitude(st,Templates,SpikeInfo,unit_column,new_label,idx=spike_id,n=3,ax=axes,id_=3)

    print(ampl,sur_ampl,sur_ampl_new,st.times[spike_id])
    plt.show()
    if abs(ampl-sur_ampl) > abs(ampl-sur_ampl_new):
        new_labels[i] = new_label
        zoom = [st.times[spike_id]-0.3*pq.s,st.times[spike_id]+0.3*pq.s]

        print(ampl,sur_ampl,sur_ampl_new,st.times[spike_id])
        fig, axes=plot_fitted_spikes(Seg, j, Models, SpikeInfo, this_unit_col, zoom=zoom, save=None,wsize=n_samples)
        plt.show()

    # if st.times[spike_id] > 10.55 and st.times[spike_id] < 10.65:
        # print(ampl,sur_ampl,amplitudes,dict_units[org_label],st.times[spike_id])

    # if ampl > sur_ampl + 0.15 and amplitudes[dict_units[new_label]] > amplitudes[dict_units[org_label]]:
    #     # print(ampl,sur_ampl,amplitudes,dict_units[org_label],st.times[spike_id])
    #     # print("Changing unit from %c to %c"%(org_label,new_label))
    #     new_labels[i] = new_label

print_msg("Num of final changes %d"%np.sum(~(SpikeInfo[unit_column]==new_labels).values))
