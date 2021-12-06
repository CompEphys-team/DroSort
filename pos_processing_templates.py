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
# mpl.rcParams['figure.dpi'] = 300

results_folder = Path(os.path.abspath(sys.argv[1]))

# results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots' / 'pos_processing' / 'templates'
os.makedirs(plots_folder, exist_ok=True)

fig_format = '.png'

Blk=get_data(sys.argv[1]+"/result.dill")
SpikeInfo = pd.read_csv(sys.argv[1]+"/SpikeInfo.csv")

unit_column = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
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
n_neighbors = 12#TODO add to config file
n_samples = Templates[:,0].size *5

neighbors_t = get_neighbors_time(Seg.analogsignals[0],st,n_samples,n_neighbors)

print_msg("Calculated window time %f based on %d neighbours"%(neighbors_t,n_neighbors))
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

# print(SpikeInfo['time'].values)
# print(np.where((SpikeInfo['time'].values > 6.5) & (SpikeInfo['time'].values < 6.9)))
# ix = SpikeInfo['id'][np.where((SpikeInfo['time'].values > 6.5) & (SpikeInfo['time'].values < 6.9))[0]]
# Templates = Templates[:,ix]

#########################################################################################################
####    get average spikes
#########################################################################################################

n_samples = np.sum(w_samples)

# Get average shapes
dt = Seg.analogsignals[0].times[1]-Seg.analogsignals[0].times[0]
dt = dt.item()
width_ms = (n_samples/10000)
mode = 'peak'

aligned_spikes = align_spikes(Templates,mode)

average_spikes = get_averages_from_units(aligned_spikes,units,SpikeInfo,unit_column)

# plot_templates(Templates, SpikeInfo, unit_column)
# plot_averages(average_spikes,SpikeInfo,unit_column)
# plt.show()

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
max_len = n_samples
# max_len = w_samples[1]

combined_templates,templates_labels = get_combined_templates([A,B],dt_c,max_len,mode)

#Plot templates
# plot_combined_templates(combined_templates,templates_labels,ncols=5)
# plt.show()

#########################################################################################################
####    compare spikes with templates
#########################################################################################################
mode = 'end'
mode = 'mean'
mode = 'neighbors'

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
    all_combined_templates=np.zeros((long_waveforms.shape[0],len(combined_templates),long_waveforms.shape[1]))

    D_pw = sp.zeros((long_waveforms_align.shape[0],aligned_templates.shape[1]))

    # for each spike
    # spikes = long_waveforms
    spikes = Templates.T
    short_size = 90
    short_aligned_spikes = aligned_spikes[:short_size,:]

    for t_i,t in enumerate(long_waveforms):
        spike_id = SpikeInfo['id'][t_i]
        # get neighbors
        # get ids from time reference 
        neighbors_ids = get_spikes_ids(t_i,neighbors_t,SpikeInfo,st)[0]

        # zoom = [st.times[spike_id]-neighbors_t*pq.s,st.times[spike_id]+neighbors_t*pq.s]
        # fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates[:short_size,:], SpikeInfo, [unit_column, unit_column], zoom=zoom, save=None,wsize=short_size)
        # plt.show()

        # get average template
        # get spikes from short template, not long version by changing short_size value
        average_spikes = get_averages_from_units(short_aligned_spikes,units,SpikeInfo.loc[neighbors_ids],unit_column,verbose=False)

        # plot_averages(average_spikes,SpikeInfo,unit_column,units)
        # plt.show()

        # get combined templates for the new averages
        all_combined_templates[t_i],templates_labels = get_combined_templates(average_spikes,dt_c,max_len,mode='end')

        # plot_combined_templates(combined_templates[t_i],templates_labels,ncols=5)
        # plt.show()

        # align both by mean
        for ct_i,ct in enumerate(all_combined_templates[t_i]):
            d_mean = (np.mean(ct)-np.mean(t))**2
            # print(d_mean)
            ct_a = align_to(ct,d_mean)
            t_a = align_to(t,d_mean)
            long_waveforms_align[t_i] = t_a
            aligned_templates[t_i,ct_i]= ct_a

        # get distances
        D_pw[t_i,:] = metrics.pairwise.euclidean_distances(aligned_templates[t_i],t.reshape(1,-1)).reshape(-1)
    distances = D_pw
else:
    # long_waveforms = np.array([align_to(np.concatenate(([t[0]]*(n_samples//2),t,[t[-1]]*(n_samples//2))),mode) for t in Templates.T])
    long_waveforms = np.array([align_to(np.concatenate((t,[t[-1]]*max_len)),mode) for t in Templates.T])
    distances=distance_to_average(long_waveforms.T,combined_templates)
print(distances.shape)


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

    # if next_peak < peak+w_time: #If spikes are close enough

    best_match = templates_labels[np.argmin(distances[t_id])]
    curr_match = [unit_titles[my_unit],unit_titles[next_unit]]

    print("guess:",best_match,distances[t_id,np.argmin(distances[t_id])])
    print("current:",curr_match)
    print(best_match != [unit_titles[my_unit],unit_titles[next_unit]])
    
    #guess different to assignation
    if best_match != curr_match:
        # try:
        # plot_combined_templates(all_combined_templates[t_id],templates_labels,ncols=5)
        # # except:
        # #     plot_combined_templates(combined_templates,templates_labels,ncols=5)

        # plt.suptitle("spike %d from unit %s"%(t_id,unit_titles[SpikeInfo[unit_column][t_id]]))

        # outpath = plots_folder / ('spike_'+str(t_id)+'_templates_grid' + fig_format)
        # plt.savefig(outpath)

        zoom = [peak- neighbors_t ,peak+neighbors_t]
        fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates, SpikeInfo, [unit_column, unit_column], zoom=zoom, save=None)
        axes[0].plot([peak,next_peak],[1,1],'.',markersize=10)
        axes[1].plot([peak,next_peak],[1,1],'.',markersize=10)
        outpath = plots_folder / ('changed_spike_'+str(t_id)+'_signal' + fig_format)
        plt.savefig(outpath)

        if best_match == ['b'] and curr_match == ['b','b']:
            print("Skiping b match",best_match)
            continue

        if best_match[0] == 'c':
            c_spikes.append(t_id)
            non_spikes.append(t_id+1)

            print("spike %d from unit %s"%(t_id,unit_titles[SpikeInfo[unit_column][t_id]]))

            # plt.show()

            # SpikeInfo[unit_column][t_id] = units[0]
            # SpikeInfo[unit_column][t_id] = units[1]
            ##TODO add new peak in SpikeInfo, Templates, etc.
        else:
            # plt.close()
            SpikeInfo[unit_column][t_id] = title_units[best_match[0]]
            # if len(best_match) < 2: #when match is only a 
            if best_match == ['a'] and next_peak < peak+w_time: #If spikes are close enough
                non_spikes.append(t_id+1)
                # SpikeInfo[unit_column][t_id+1] = '-1'
            # elif best_match >:
            #     print(SpikeInfo[unit_column][t_id+1])
            #     print(title_units[best_match[1]])
            #     SpikeInfo[unit_column][t_id+1] = title_units[best_match[1]]

print(c_spikes,non_spikes)
        # else:
        #     plt.close()

#Change c_spikes and non_spikes

#Remove spikes from templates
#Get new spikes 

