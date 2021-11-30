import matplotlib.pyplot as plt
import pandas as pd
from sssio import * 
from plotters import *
from functions import *
from sys import path
from superpos_functions import align_spike

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
fig_format = '.png'

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
    # spike = align_to(spike, mode,dt,width_ms)
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


outpath = results_folder / 'template_a.npy'
sp.save(outpath, A)
outpath = results_folder / 'template_b.npy'
sp.save(outpath, B)

combined_templates = []

mode = 'end'
dt_c = 2
# ext_size = w_samples[1]
ext_size = n_samples//2
max_len = w_samples[1]
#Add AB templates
combine_templates(combined_templates,A,B,dt_c,max_len,mode)
n = len(combined_templates)
templates_labels = [['a','b']]*n

#Add BA templates
combine_templates(combined_templates,B,A,dt_c,max_len,mode)
n = len(combined_templates) -n
templates_labels+=[['b','a']]*n

#Add sum of two
comb_t =np.array(np.concatenate((A,[A[-1]]*max_len))+np.concatenate((B,[B[-1]]*max_len)))
# comb_t =np.array(np.concatenate(([A[0]]*(n_samples//2),A,[A[-1]]*(n_samples//2)))+np.concatenate(([B[0]]*(n_samples//2),B,[B[-1]]*(n_samples//2))))
combined_templates.append(np.array(align_to(comb_t,mode)))
templates_labels.append(['c'])

#Add simple A
comb_t =np.array(np.concatenate((A,[A[-1]]*max_len)))
# comb_t =np.array(np.concatenate(([A[0]]*(n_samples//2),A,[A[-1]]*(n_samples//2))))
combined_templates.append(np.array(align_to(comb_t,mode)))
templates_labels.append(['a'])

#Add simple B
comb_t =np.array(np.concatenate((B,[B[-1]]*max_len)))
# comb_t =np.array(np.concatenate(([B[0]]*(n_samples//2),B,[B[-1]]*(n_samples//2))))
combined_templates.append(np.array(align_to(comb_t,mode)))
templates_labels.append(['b'])

#Plot templates
ncols = 5 
nrows = int(np.ceil(len(combined_templates)/ncols))
print(nrows,ncols)

fig, axes= plt.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True,figsize=(ncols*3,nrows*2))

print(ncols,nrows)
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

# exit()

#########################################################################################################
####    compare spikes with templates
#########################################################################################################
mode = 'end'

#Get distances

#Mode 1 align each combinatio to the mean and computes distances
if mode == 'mean':
    #TODO align by mean
    long_waveforms = np.array([np.concatenate(([t[0]]*(n_samples//2),t,[t[-1]]*(n_samples//2))) for t in Templates.T])

    aligned_templates=np.zeros((long_waveforms.shape[0],len(combined_templates),long_waveforms.shape[1]))
    long_waveforms_align = np.zeros(long_waveforms.shape)

    D_pw = sp.zeros((long_waveforms_align.shape[0],aligned_templates.shape[1]))
    print(aligned_templates.shape)
    for t_i,t in enumerate(long_waveforms):
        for ct_i,ct in enumerate(combined_templates):
            d_mean = (np.mean(ct)-np.mean(t))**2
            # print(d_mean)
            ct_a = align_to(ct,d_mean)
            t_a = align_to(t,d_mean)
            long_waveforms_align[t_i] = t_a
            aligned_templates[t_i,ct_i]= ct_a
        D_pw[t_i,:] = metrics.pairwise.euclidean_distances(aligned_templates[t_i],t.reshape(1,-1)).reshape(-1)
 
    distances = D_pw

else:
    # long_waveforms = np.array([align_to(np.concatenate(([t[0]]*(n_samples//2),t,[t[-1]]*(n_samples//2))),mode) for t in Templates.T])
    long_waveforms = np.array([align_to(np.concatenate((t,[t[-1]]*max_len)),mode) for t in Templates.T])
    distances=distance_to_average(long_waveforms.T,combined_templates)
print(distances.shape)

isis = [ b-a for a,b in zip(SpikeInfo['time'][:-1],SpikeInfo['time'][1:])]

plt.hist(isis,bins=3,width=0.2)
plt.show()


#Compare all templates
colors = get_colors(units)

t_colors = [colors[unit] for unit in SpikeInfo[unit_column]]

c_spikes = []

for t_id,t in enumerate(long_waveforms):
    #Get this peak and next one
    peak = SpikeInfo['time'][t_id] 
    next_peak = SpikeInfo['time'][t_id+1]

    my_unit = SpikeInfo[unit_column][t_id]
    next_unit = SpikeInfo[unit_column][t_id+1]

    # print(next_peak,peak+(n_samples//2)/10000)

    #Reasonable distance to next spike
    w_time = (n_samples//2)/10000

    if next_peak < peak+w_time: #If spikes are close enough

        best_match = templates_labels[np.argmin(distances[t_id])]

        print("guess:",best_match,distances[t_id,np.argmin(distances[t_id])])
        print("current:",[unit_titles[my_unit],unit_titles[next_unit]])
        print(best_match != [unit_titles[my_unit],unit_titles[next_unit]])
      
        fig, axes= plt.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True,figsize=(ncols*4,nrows*2))

        for c,ct in enumerate(combined_templates):
            i,j = c//ncols,c%ncols
            axes[i,j].plot(ct)
            axes[i,j].plot(t,color=t_colors[t_id])
            axes[i,j].text(0.25, 0.75,"%.3f"%distances[t_id,c])

        plt.suptitle("spike %d from unit %s"%(t_id,unit_titles[SpikeInfo[unit_column][t_id]]))
        
        #guess different to assignation
        if best_match != [unit_titles[my_unit],unit_titles[next_unit]]:
            outpath = plots_folder / ('spike_'+str(t_id)+'_templates_grid' + fig_format)
            plt.savefig(outpath)

            zoom = [peak-0.3,peak+0.3]
            fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, [unit_column, unit_column], zoom=zoom, save=None)
            axes[0].plot([peak,next_peak],[1,1],'.',markersize=10)
            axes[1].plot([peak,next_peak],[1,1],'.',markersize=10)
            outpath = plots_folder / ('changed_spike_'+str(t_id)+'_signal' + fig_format)
            plt.savefig(outpath)

            if best_match[0] == 'c':
                c_spikes.append(t_id)
                print("spike %d from unit %s"%(t_id,unit_titles[SpikeInfo[unit_column][t_id]]))

                plt.show()

                # SpikeInfo[unit_column][t_id] = units[0]
                # SpikeInfo[unit_column][t_id] = units[1]
                ##TODO add new peak in SpikeInfo, Templates, etc.
            else:
                plt.close()
                SpikeInfo[unit_column][t_id] = title_units[best_match[0]]
                if len(best_match) < 2:
                    SpikeInfo[unit_column][t_id+1] = '-1'
                else:
                    SpikeInfo[unit_column][t_id+1] = title_units[best_match[1]]

        else:
            plt.close()


#Remove spikes from templates
#Get new spikes 



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