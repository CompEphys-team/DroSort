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

# spike_ids = SpikeInfo['id'].values    
# dict_units = {u:i for i,u in enumerate(units)}

st = Blk.segments[0].spiketrains[0]
Seg = Blk.segments[0]


# new_labels = copy.deepcopy(SpikeInfo[unit_column].values)
n_neighbors = 3 #TODO add to config file
n_samples = Templates[:,0].size *5

#Time of one spike * number of spikes * 2 (approximate isis)
neighbors_t = (n_samples/1000)*n_neighbors 
print(neighbors_t)
# #new column in SpikeInfo with changes
# SpikeInfo['unit_pospro'] = new_labels
# ids = []

# for i, (spike_id,org_label) in enumerate(zip(spike_ids,SpikeInfo[unit_column])):
#     if org_label == '-1':
#         continue

#     spike = Templates[:, spike_id].T
#     # ampl = max(spike)-min(spike)
#     ampl = max(spike)-min(spike[spike.size//2:]) #half spike to avoid noise

#     new_label = units[(dict_units[org_label]+1)%2]

#     sur_ampl = get_neighbors_amplitude(st,Templates,SpikeInfo,unit_column,org_label,idx=spike_id,t=neighbors_t)
#     sur_ampl_new = get_neighbors_amplitude(st,Templates,SpikeInfo,unit_column,new_label,idx=spike_id,t=neighbors_t)

#     dur = get_duration(spike)
#     sur_dur = get_neighbors_amplitude(st,Templates,SpikeInfo,unit_column,org_label,idx=spike_id,t=neighbors_t)
#     sur_dur_new = get_neighbors_amplitude(st,Templates,SpikeInfo,unit_column,new_label,idx=spike_id,t=neighbors_t)

#     # if abs(ampl-sur_ampl) > abs(ampl-sur_ampl_new)+0.05 and abs(dur-sur_dur) > abs(dur-sur_dur_new):
#     if abs(ampl-sur_ampl) > abs(ampl-sur_ampl_new):
#         SpikeInfo['unit_pospro'][i] = new_label
#         new_labels[i] = new_label
#         ids.append(i)

#         # if st.times[spike_id] > 6.2 and st.times[spike_id] < 6.5:
#         # zoom = [st.times[spike_id]-neighbors_t*pq.s,st.times[spike_id]+neighbors_t*pq.s]
#         # print(ampl,sur_ampl,sur_ampl_new,st.times[spike_id])
#         # print(ampl,abs(ampl-sur_ampl),abs(ampl-sur_ampl_new),SpikeInfo[unit_column][i],SpikeInfo['unit_pospro'][i])
#         # print(dur,sur_dur,sur_dur_new,st.times[spike_id])
#         # print(dur,abs(dur-sur_dur),abs(dur-sur_ampl_new),SpikeInfo[unit_column][i],SpikeInfo['unit_pospro'][i])
#         # fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, [unit_column, 'unit_pospro'], zoom=zoom, save=None,wsize=n_samples)

#         # plt.show()  

# print_msg("Num of final changes %d"%np.sum(~(SpikeInfo[unit_column]==new_labels).values))

# # max_window=1.5

# # for j, Seg in enumerate(Blk.segments):
# #     seg_name = Path(Seg.annotations['filename']).stem

# #     asig = Seg.analogsignals[0]
# #     max_window = int(max_window*asig.sampling_rate) #FIX conversion from secs to points
# #     n_plots = asig.shape[0]//max_window

# #     for n_plot in range(0,n_plots):
# #         # outpath = plots_folder / (seg_name + '_fitted_spikes%s_%s_%d'%(extension,max_window,n_plot) + fig_format)
# #         ini = n_plot*max_window + max_window
# #         end = ini + max_window
# #         end = min(end, Seg.analogsignals[0].shape[0])
# #         zoom = [ini,end]/asig.sampling_rate

# #         plot_fitted_spikes(Seg, j, Templates, SpikeInfo, unit_column, zoom=zoom, save=None,wsize=n_samples)

# # plt.show()
# print(ids)
# Blk = populate_block(Blk,SpikeInfo,'unit_pospro',units)
# st = Blk.segments[0].spiketrains[0]

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


print_msg(' - getting templates - ')

fs = Blk.segments[0].analogsignals[0].sampling_rate
# n_samples = (wsize * fs).simplified.magnitude.astype('int32')

w_samples = (20,70)

templates = []
for j, seg in enumerate(Blk.segments):
    data = seg.analogsignals[0].magnitude.flatten()
    inds = (seg.spiketrains[0].times * fs).simplified.magnitude.astype('int32')

    templates.append(get_Templates(data, inds, w_samples))

Templates = sp.concatenate(templates,axis=1)
print(Templates.shape)

# print(SpikeInfo['time'].values)
# print(np.where((SpikeInfo['time'].values > 6.5) & (SpikeInfo['time'].values < 6.9)))
# ix = SpikeInfo['id'][np.where((SpikeInfo['time'].values > 6.5) & (SpikeInfo['time'].values < 6.9))[0]]
# Templates = Templates[:,ix]

print(Templates.shape)


# Templates = np.array([t[t.size//2-10:t.size//2+10] for t in Templates.T]).T

#########################################################################################################
####    get average spikes
#########################################################################################################

n_samples = np.sum(w_samples)

# Get average shapes
dt = Seg.analogsignals[0].times[1]-Seg.analogsignals[0].times[0]
dt = dt.item()
width_ms = (n_samples/1000)
mode = 'peak'

### align spikes

average_spikes = []
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
    ix = SpikeInfo.groupby(last_unit_col).get_group(unit)['id']
    # ix_u = SpikeInfo.groupby('unit_pospro').get_group(unit)['id']
    # ix = SpikeInfo['id'][np.where((SpikeInfo['time'][ix] > 6.5) & (SpikeInfo['time'][ix] < 6.9))[0]]
    # ix = SpikeInfo['id'][np.where((SpikeInfo['time'][ix] < 1))[0]]

    # print(len(ix))

    templates = aligned_spikes[:,ix].T

    print_msg("Averaging %d spikes for cluster %c"%(len(templates),unit))

    average_spike = np.average(templates, axis=0)

    average_spikes.append(average_spike)

# average_spikes=np.array(average_spikes)

plot_templates(Templates, SpikeInfo, last_unit_col)
# plot_averages(average_spikes,SpikeInfo,last_unit_col)
# plt.show()

# for i,spike in enumerate(Templates[:,ix].T):
#     spike = align_spike(spike, width_ms,dt,spike_i,mode)
#     plot_averages_with_spike(spike,average_spikes,SpikeInfo,'unit_pospro',SpikeInfo['unit_pospro'][i])
#     plt.show()


#########################################################################################################
####    get template combinations
#########################################################################################################

amps = [max(av)-min(av) for av in average_spikes]

a_id = np.argmax(amps)
b_id = np.argmin(amps)

A = average_spikes[a_id]
B = average_spikes[b_id]

titles={a_id:'a',b_id:'b'}
unit_titles={units[a_id]:'a',units[b_id]:'b'}
title_units = {'a':units[a_id],'b':units[b_id]}

print(titles)
print(unit_titles)
plot_averages(average_spikes,SpikeInfo,last_unit_col,title=titles)
plt.show()


combined_templates = []

mode = 'end'
dt_c = 2
#Add AB templates
combine_templates(combined_templates,A,B,dt_c,w_samples,mode)
n = len(combined_templates)
templates_labels = [['a','b']]*n

#Add BA templates
combine_templates(combined_templates,B,A,dt_c,w_samples,mode)
n = len(combined_templates) -n
templates_labels+=[['b','a']]*n

#Add sum of two
comb_t =np.array(np.concatenate(([A[0]]*(n_samples//2),A,[A[-1]]*(n_samples//2)))+np.concatenate(([B[0]]*(n_samples//2),B,[B[-1]]*(n_samples//2))))
combined_templates.append(np.array(align_to(comb_t,mode)))
templates_labels.append(['c'])

#Add simple A
comb_t =np.array(np.concatenate(([A[0]]*(n_samples//2),A,[A[-1]]*(n_samples//2))))
combined_templates.append(np.array(align_to(comb_t,mode)))
templates_labels.append(['a'])

#Add simple B
comb_t =np.array(np.concatenate(([B[0]]*(n_samples//2),B,[B[-1]]*(n_samples//2))))
combined_templates.append(np.array(align_to(comb_t,mode)))
templates_labels.append(['b'])

ncols = 9
nrows = round(len(combined_templates)/ncols)
print(nrows,ncols)

fig, axes= plt.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True,figsize=(ncols*3,nrows*2))

print(len(templates_labels))
print(len(combined_templates))
for c,ct in enumerate(combined_templates):
    i,j = c//ncols,c%ncols

    axes[i,j].plot(ct)

    peak_inds = signal.argrelmax(ct)[0]
    peak_inds = peak_inds[np.argsort(ct[peak_inds])[-len(templates_labels[c]):]]

    axes[i,j].plot(peak_inds,ct[peak_inds],'.')
    axes[i,j].text(0.25, 0.75,str(templates_labels[c]))
    # plt.ylim(-2,0)

plt.tight_layout()
plt.show()

#########################################################################################################
####    compare spikes with templates
#########################################################################################################
#TODO align by mean
# for long_templates
#     for ct in combined_templates:
#         d_mean = abs(np.mean(ct)-np.mean(long_templates))
#         ct_a = align_to(ct,d_mean)



long_templates = np.array([align_to(np.concatenate(([t[0]]*(n_samples//2),t,[t[-1]]*(n_samples//2))),mode) for t in Templates.T])
distances=distance_to_average(long_templates.T,units,combined_templates)
print(distances.shape)

colors = get_colors(units)

t_colors = [colors[unit] for unit in SpikeInfo[unit_column]]

c_spikes = []

for t_id,t in enumerate(long_templates):
    peak = SpikeInfo['time'][t_id] 
    next_peak = SpikeInfo['time'][t_id+1]

    my_unit = SpikeInfo[unit_column][t_id]
    next_unit = SpikeInfo[unit_column][t_id+1]
    # print(next_peak,peak+(n_samples//2)/10000)
    w_time = (n_samples//2)/10000
    if next_peak < peak+w_time: #If close spikes

        best_match = templates_labels[np.argmin(distances[t_id])]
        print("guess:",best_match,distances[t_id,np.argmin(distances[t_id])])
        print("current:",[unit_titles[my_unit],unit_titles[next_unit]])
        print(best_match != [unit_titles[my_unit],unit_titles[next_unit]])
      
        fig, axes= plt.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True,figsize=(ncols*2,nrows*2))

        for c,ct in enumerate(combined_templates):
            i,j = c//ncols,c%ncols
            axes[i,j].plot(ct)
            axes[i,j].plot(t,color=t_colors[t_id])
            axes[i,j].text(0.25, 0.75,"%.3f"%distances[t_id,c])

        plt.suptitle("spike %d from unit %s"%(t_id,unit_titles[SpikeInfo[unit_column][t_id]]))

        zoom = [peak-0.3,peak+0.3]
        fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, [unit_column, unit_column], zoom=zoom, save=None)
        axes[0].plot([peak,next_peak],[1,1],'.',markersize=10)
        axes[1].plot([peak,next_peak],[1,1],'.',markersize=10)
        
        plt.show()

        if best_match != [unit_titles[my_unit],unit_titles[next_unit]]:
            if best_match[0] == 'c':
                c_spikes.append(t_id)
                # SpikeInfo[unit_column][t_id] = units[0]
                # SpikeInfo[unit_column][t_id] = units[1]
                ##TODO add new peak in SpikeInfo, Templates, etc.
            else:
                SpikeInfo[unit_column][t_id] = title_units[best_match[0]]
                if len(best_match) < 2:
                    SpikeInfo[unit_column][t_id+1] = '-1'
                else:
                    SpikeInfo[unit_column][t_id+1] = title_units[best_match[1]]

        


#########################################################################################################
####    compare spikes with average
#########################################################################################################


# units = get_units(SpikeInfo,last_unit_col)
# # distances = distance_to_average(Templates,units,average_spikes)
# distances = distance_to_average(aligned_spikes,units,average_spikes)

# dict_units = {u:i for i,u in enumerate(units)}

# new_labels = copy.deepcopy(SpikeInfo[last_unit_col].values)

# SpikeInfo['unit_pospro2'] = new_labels

# print(units)

# for i,d in enumerate(distances):
#     org_unit = SpikeInfo[last_unit_col][i]
#     try:
#         unit_id = dict_units[org_unit]
#     except KeyError:
#         continue

#     new_label = units[(unit_id+1)%2]

#     if d[unit_id] > d[(unit_id+1)%2]+0.1: #if distance to the other cluster is bigger
#         SpikeInfo['unit_pospro2'][i] = new_label

#     if SpikeInfo['time'][i] >10 and SpikeInfo['time'][i] < 12:
#         print("my unit","other unit",SpikeInfo['time'][i])
#         print(d[unit_id],d[(unit_id+1)%2])

#         zoom = [st.times[i]-neighbors_t*pq.s,st.times[i]+neighbors_t*pq.s]

#         fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, [last_unit_col, 'unit_pospro2'], zoom=zoom, save=None,wsize=30)

#         plot_averages_with_spike(aligned_spikes[:,i],average_spikes,SpikeInfo,last_unit_col,org_unit)
#         plt.show()



# print_msg("Changes by shape: %d"%(np.sum(~(SpikeInfo[last_unit_col]==SpikeInfo['unit_pospro2']))))


# max_window = 0.3
# extension1 = 'pospro1'
# extension2 = 'pospro2'
# fig_format='.png'

# # plot_fitted_spikes_complete(Blk, Templates, SpikeInfo, [unit_column, last_unit_col], max_window, plots_folder, '.png',wsize=n_samples,extension='_pospro1',plot_function=plot_compared_fitted_spikes)
# # plot_fitted_spikes_complete(Blk, Templates, SpikeInfo, [last_unit_col, 'unit_pospro2'], max_window, plots_folder, '.png',wsize=n_samples,extension='_pospro2',plot_function=plot_compared_fitted_spikes)



# # ix = SpikeInfo.groupby(['good']).get_group((True))['id']
# # for spike in Templates[:,ix].T:
# #     plt.plot(spike)
# #     plt.show()