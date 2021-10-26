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

results_folder = Path(os.path.abspath(sys.argv[1]))

print(results_folder)
# results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots'
print(plots_folder)


Blk=get_data(sys.argv[1]+"/result.dill")

SpikeInfo = pd.read_csv(sys.argv[1]+"/SpikeInfo.csv")

print(SpikeInfo.keys())

last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
unit_column = last_unit_col
SpikeInfo = SpikeInfo.astype({last_unit_col: str})
units = get_units(SpikeInfo,unit_column)

Templates= np.load(sys.argv[1]+"/Templates.npy")



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
        # fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, [unit_column, 'unit_pospro'], zoom=zoom, save=None,wsize=n_samples)

        # plt.show()  

print_msg("Num of final changes %d"%np.sum(~(SpikeInfo[unit_column]==new_labels).values))

# max_window=1.5

# for j, Seg in enumerate(Blk.segments):
#     seg_name = Path(Seg.annotations['filename']).stem

#     asig = Seg.analogsignals[0]
#     max_window = int(max_window*asig.sampling_rate) #FIX conversion from secs to points
#     n_plots = asig.shape[0]//max_window

#     for n_plot in range(0,n_plots):
#         # outpath = plots_folder / (seg_name + '_fitted_spikes%s_%s_%d'%(extension,max_window,n_plot) + fig_format)
#         ini = n_plot*max_window + max_window
#         end = ini + max_window
#         end = min(end, Seg.analogsignals[0].shape[0])
#         zoom = [ini,end]/asig.sampling_rate

#         plot_fitted_spikes(Seg, j, Templates, SpikeInfo, unit_column, zoom=zoom, save=None,wsize=n_samples)

# plt.show()
print(ids)
Blk = populate_block(Blk,SpikeInfo,'unit_pospro',units)
st = Blk.segments[0].spiketrains[0]

# for i, (spike_id,org_label) in enumerate(zip(spike_ids,SpikeInfo['unit_pospro'])):
#     if i in ids:
#         zoom = [st.times[spike_id]-neighbors_t*pq.s,st.times[spike_id]+neighbors_t*pq.s]
#         print(ampl,sur_ampl,sur_ampl_new,st.times[spike_id])
#         fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, [unit_column, 'unit_pospro'], zoom=zoom, save=None,wsize=n_samples)
#         axes[0].plot(st.times[spike_id],1,'.')
#         axes[1].plot(st.times[spike_id],1,'.')
#         plt.show()



################################################################
##  
##              Unassign by shapes
##  
################################################################
######WARNING BAD CLUSTERING
#TODO fix measure of distance --> bad reassignation. 


# print(SpikeInfo['time'].values)
# print(np.where((SpikeInfo['time'].values > 6.5) & (SpikeInfo['time'].values < 6.9)))
# ix = SpikeInfo['id'][np.where((SpikeInfo['time'].values > 6.5) & (SpikeInfo['time'].values < 6.9))[0]]
# Templates = Templates[:,ix]

print(Templates.shape)


# Templates = np.array([t[t.size//2-10:t.size//2+10] for t in Templates.T]).T

# Get average shapes
dt = Seg.analogsignals[0].times[1]-Seg.analogsignals[0].times[0]
dt = dt.item()
width_ms = (n_samples/1000)
mode = 'min'

average_spikes = []
# aligned_all_spikes = []
aligned_spikes = []
for spike_i,spike in enumerate(Templates.T):
    spike = align_spike(spike, width_ms,dt,spike_i,mode)
    if spike == []:
        continue

    if spike != []:
        aligned_spikes.append(spike)

aligned_spikes = np.array(aligned_spikes).T

print(aligned_spikes.shape)
print(Templates.shape)

for unit in units:
    ix = SpikeInfo.groupby('unit_pospro').get_group(unit)['id']
    # ix_u = SpikeInfo.groupby('unit_pospro').get_group(unit)['id']
    # ix = SpikeInfo['id'][np.where((SpikeInfo['time'][ix] > 6.5) & (SpikeInfo['time'][ix] < 6.9))[0]]
    # ix = SpikeInfo['id'][np.where((SpikeInfo['time'][ix] < 1))[0]]

    # print(len(ix))

    templates = aligned_spikes[:,ix].T

    print_msg("Averaging %d spikes for cluster %c"%(len(templates),unit))

    average_spike = np.average(templates, axis=0)

    average_spikes.append(average_spike)

# average_spikes=np.array(average_spikes)

plot_templates(Templates, SpikeInfo, 'unit_pospro')
plot_averages(average_spikes,SpikeInfo,'unit_pospro')
plt.show()

for i,spike in enumerate(Templates[:,ix].T):
    spike = align_spike(spike, width_ms,dt,spike_i,mode)
    plot_averages_with_spike(spike,average_spikes,SpikeInfo,'unit_pospro',SpikeInfo['unit_pospro'][i])
    plt.show()



units = get_units(SpikeInfo,'unit_pospro')
# distances = distance_to_average(Templates,average_spikes)
distances = distance_to_average(aligned_spikes,average_spikes)

dict_units = {u:i for i,u in enumerate(units)}

new_labels = copy.deepcopy(SpikeInfo['unit_pospro'].values)

SpikeInfo['unit_pospro2'] = new_labels

print(units)

for i,d in enumerate(distances):
    org_unit = SpikeInfo['unit_pospro'][i]
    try:
        unit_id = dict_units[org_unit]
    except KeyError:
        continue

    new_label = units[(unit_id+1)%2]

    if d[unit_id] > d[(unit_id+1)%2]+0.1: #if distance to the other cluster is bigger
        SpikeInfo['unit_pospro2'][i] = new_label

    # if SpikeInfo['time'][i] > 6.6 and SpikeInfo['time'][i] < 6.9:
    # print("my unit","other unit",SpikeInfo['time'][i])
    # print(d[unit_id],d[(unit_id+1)%2])

    # zoom = [st.times[i]-neighbors_t*pq.s,st.times[i]+neighbors_t*pq.s]

    # fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, ['unit_pospro', 'unit_pospro2'], zoom=zoom, save=None,wsize=n_samples)

    # plot_averages_with_spike(aligned_spikes[:,i],average_spikes,SpikeInfo,'unit_pospro',org_unit)
    # plt.show()



print_msg("Changes by shape: %d"%(np.sum(~(SpikeInfo['unit_pospro']==SpikeInfo['unit_pospro2']))))


max_window = 0.3
extension1 = 'pospro1'
extension2 = 'pospro2'
fig_format='.png'

# plot_fitted_spikes_complete(Blk, Templates, SpikeInfo, [unit_column, 'unit_pospro'], max_window, plots_folder, '.png',wsize=n_samples,extension='_pospro1',plot_function=plot_compared_fitted_spikes)
# plot_fitted_spikes_complete(Blk, Templates, SpikeInfo, ['unit_pospro', 'unit_pospro2'], max_window, plots_folder, '.png',wsize=n_samples,extension='_pospro2',plot_function=plot_compared_fitted_spikes)



ix = SpikeInfo.groupby(['good']).get_group((True))['id']
for spike in Templates[:,ix].T:
    plt.plot(spike)
    plt.show()