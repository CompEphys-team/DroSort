import matplotlib.pyplot as plt
import pandas as pd
from sssio import * 
from plotters import *
from functions import *
from sys import path
from superpos_functions import align_spike
from posprocessing_functions import *

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
mpl.rcParams['figure.dpi'] = 300

results_folder = Path(os.path.abspath(sys.argv[1]))

print(results_folder)
# results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots'
print(plots_folder)
fig_format = '.png'

Blk=get_data(sys.argv[1]+"/result.dill")

SpikeInfo = pd.read_csv(sys.argv[1]+"/SpikeInfo.csv")

print(SpikeInfo.keys())

unit_column = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
# unit_column = last_unit_col
SpikeInfo = SpikeInfo.astype({unit_column: str})
units = get_units(SpikeInfo,unit_column)
Templates= np.load(sys.argv[1]+"/Templates_ini.npy")


#third cluster fix
if '-2' in units:
    units.remove('-2') 
if len(units) > 2:
    print_msg("Too many units: %s. Try run cluster_identification.py first"%units)
    exit()


print_msg("Number of spikes in trace: %d"%SpikeInfo[unit_column].size)
print_msg("Number of good spikes: %d"%len(SpikeInfo.groupby(['good']).get_group(True)[unit_column]))
# print_msg("Number of bad spikes: %d"%len(SpikeInfo.groupby(['good']).get_group(False)[unit_column]))
print_msg("Number of clusters: %d"%len(units))


st = Blk.segments[0].spiketrains[0]
Seg = Blk.segments[0]


# new_labels = copy.deepcopy(SpikeInfo[unit_column].values)
n_neighbors = 6 #TODO add to config file
n_samples = Templates[:,0].size *5

# #Time of one spike * number of spikes * 2 (approximate isis)
# neighbors_t = (n_samples/1000)*n_neighbors 
# print(neighbors_t)

neighbors_t = get_neighbors_time(Seg.analogsignals[0],st,n_samples,n_neighbors)

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
width_ms = (n_samples/10000)
mode = 'peak'

### align spikes

# average_spikes = []
# aligned_spikes = []
# for spike_i,spike in enumerate(Templates.T):
#     spike = align_spike(spike, width_ms,dt,spike_i,mode)
#     # spike = align_to(spike, mode,dt,width_ms)
#     if spike == []:
#         continue

#     if spike != []:
#         aligned_spikes.append(spike)
# aligned_spikes = np.array(aligned_spikes).T

# aligned_spikes = np.array([align_to(spike, mode,dt,width_ms) for spike in Templates.T]).T
aligned_spikes = align_spikes(Templates,mode)

print(aligned_spikes.shape)
print(Templates.shape)

average_spikes = get_averages_from_units(aligned_spikes,units,SpikeInfo,unit_column)

# plot_templates(Templates, SpikeInfo, unit_column)
# plot_averages(average_spikes,SpikeInfo,unit_column)
# plt.show()

# exit()
#########################################################################################################
####    get template combinations
#########################################################################################################

#label spike clusters
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
# plot_averages(average_spikes,SpikeInfo,unit_column,title=titles)
# plt.show()


outpath = results_folder / 'template_a.npy'
sp.save(outpath, A)
outpath = results_folder / 'template_b.npy'
sp.save(outpath, B)

# combined_templates = []

##get_combined_templates
mode = 'end'
dt_c = 2
# ext_size = w_samples[1]
# ext_size = n_samples//2
max_len = n_samples
# max_len = w_samples[1]

print(A.shape, B.shape)
print(max_len)

combined_templates,templates_labels = get_combined_templates([A,B],dt_c,max_len,mode)

#Plot templates
# plot_combined_templates(combined_templates,templates_labels,ncols=5)
# plt.show()

# exit()

#########################################################################################################
####    compare spikes with templates
#########################################################################################################
mode = 'end'
# mode = 'mean'

#Get distances

#Mode 1 align each combination to the mean and computes distances
if mode == 'mean':
    long_waveforms = np.array([np.concatenate(([t[0]]*(n_samples//2),t,[t[-1]]*(n_samples//2))) for t in Templates.T])
    aligned_templates=np.zeros((long_waveforms.shape[0],len(combined_templates),long_waveforms.shape[1]))
    long_waveforms_align = np.zeros(long_waveforms.shape)

    D_pw = sp.zeros((long_waveforms_align.shape[0],aligned_templates.shape[1]))

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

elif mode == 'neighbors':
    long_waveforms = np.array([np.concatenate(([t[0]]*(n_samples//2),t,[t[-1]]*(n_samples//2))) for t in Templates.T])
    aligned_templates=np.zeros((long_waveforms.shape[0],len(combined_templates),long_waveforms.shape[1]))
    long_waveforms_align = np.zeros(long_waveforms.shape)

    D_pw = sp.zeros((long_waveforms_align.shape[0],aligned_templates.shape[1]))

    # for each spike
    # spikes = long_waveforms
    spikes = Templates.T
    short_size = 90
    short_aligned_spikes = aligned_spikes[:short_size,:]
    for t_i,t in enumerate(spikes):
        spike_id = SpikeInfo['id'][t_i]
        # get neighbors
        # get ids from time reference 
        neighbors_ids = get_spikes_ids(t_i,neighbors_t,SpikeInfo,st)[0]

        zoom = [st.times[spike_id]-neighbors_t*pq.s,st.times[spike_id]+neighbors_t*pq.s]
        # fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, [unit_column, unit_column], zoom=zoom, save=None,wsize=n_samples)
        fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates[:short_size,:], SpikeInfo, [unit_column, unit_column], zoom=zoom, save=None,wsize=short_size)
        plt.show()

        # get long_templates from id 

        # long_neighbors = spikes[neighbors_ids,:]
        # plt.plot(short_aligned_spikes[:,neighbors_ids])
        # plt.show()


        # get average template
        # aligned_waveforms = align_spikes(spikes,mode='peak')
        # print(SpikeInfo.loc[neighbors_ids])
        #TODO: get spikes from short template, not long version
        average_spikes = get_averages_from_units(short_aligned_spikes,units,SpikeInfo.loc[neighbors_ids],unit_column)

        plot_averages(average_spikes,SpikeInfo,unit_column,units)
        plt.show()

        # get combined templates for the new averages
        combined_templates[t_i],templates_labels = get_combined_templates(average_spikes,dt_c,max_len,mode='end')

        plot_combined_templates(combined_templates[t_i],templates_labels,ncols=5)
        plt.show()

        # align both by mean
        for ct_i,ct in enumerate(combined_templates):
            d_mean = (np.mean(ct)-np.mean(t))**2
            # print(d_mean)
            ct_a = align_to(ct,d_mean)
            t_a = align_to(t,d_mean)
            long_waveforms_align[t_i] = t_a
            aligned_templates[t_i,ct_i]= ct_a

        # get distances
        D_pw[t_i,:] = metrics.pairwise.euclidean_distances(aligned_templates[t_i],t.reshape(1,-1)).reshape(-1)

else:
    # long_waveforms = np.array([align_to(np.concatenate(([t[0]]*(n_samples//2),t,[t[-1]]*(n_samples//2))),mode) for t in Templates.T])
    long_waveforms = np.array([align_to(np.concatenate((t,[t[-1]]*max_len)),mode) for t in Templates.T])
    distances=distance_to_average(long_waveforms.T,combined_templates)
print(distances.shape)

# isis = [ b-a for a,b in zip(SpikeInfo['time'][:-1],SpikeInfo['time'][1:])]

# plt.hist(isis,bins=3,width=0.2)
# plt.show()


#Compare all templates
colors = get_colors(units)

#TODO: fix no need to assign color to -1 and -2...
colors['-1'] = 'k'
colors['-2'] = 'k'
# print(colors)

t_colors = [colors[unit] for unit in SpikeInfo[unit_column]]

c_spikes = []
non_spikes = []

for t_id,t in enumerate(long_waveforms):
    #Get this peak and next one
    try:
        peak = SpikeInfo['time'][t_id] 
        next_peak = SpikeInfo['time'][t_id+1]
    except:
        continue
    my_unit = SpikeInfo[unit_column][t_id]
    next_unit = SpikeInfo[unit_column][t_id+1]

    # print(my_unit,next_unit)

    # TODO: fix -1 spikes!!!
    if int(my_unit) < 0 or int(next_unit) < 0:
        continue
        # zoom = [peak-0.3,peak+0.3]
        # fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, [unit_column, unit_column], zoom=zoom, save=None)
        # axes[0].plot([peak,next_peak],[1,1],'.',markersize=10)
        # axes[1].plot([peak,next_peak],[1,1],'.',markersize=10)
        # plt.show()
    # print(next_peak,peak+(n_samples//2)/10000)

    #Reasonable distance to next spike
    w_time = (n_samples//2)/10000

    if next_peak < peak+w_time: #If spikes are close enough

        best_match = templates_labels[np.argmin(distances[t_id])]

        print("guess:",best_match,distances[t_id,np.argmin(distances[t_id])])
        print("current:",[unit_titles[my_unit],unit_titles[next_unit]])
        print(best_match != [unit_titles[my_unit],unit_titles[next_unit]])
      
        # fig, axes= plt.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True,figsize=(ncols*4,nrows*2))

        # for c,ct in enumerate(combined_templates):
        #     i,j = c//ncols,c%ncols
        #     axes[i,j].plot(ct)
        #     axes[i,j].plot(t,color=t_colors[t_id])
        #     axes[i,j].text(0.25, 0.75,"%.3f"%distances[t_id,c])

        plot_combined_templates(combined_templates,templates_labels,ncols=5)

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
                non_spikes.append(t_id+1)

                print("spike %d from unit %s"%(t_id,unit_titles[SpikeInfo[unit_column][t_id]]))

                # plt.show()

                # SpikeInfo[unit_column][t_id] = units[0]
                # SpikeInfo[unit_column][t_id] = units[1]
                ##TODO add new peak in SpikeInfo, Templates, etc.
            else:
                plt.close()
                SpikeInfo[unit_column][t_id] = title_units[best_match[0]]
                if len(best_match) < 2:
                    non_spikes.append(t_id+1)
                    # SpikeInfo[unit_column][t_id+1] = '-1'
                else:
                    SpikeInfo[unit_column][t_id+1] = title_units[best_match[1]]

        else:
            plt.close()

#Change c_spikes and non_spikes

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