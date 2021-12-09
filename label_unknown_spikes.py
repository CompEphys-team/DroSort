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

# results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots' / 'pos_processing' / 'label_unknown'
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
n_neighbors = 15#TODO add to config file
n_samples = Templates[:,0].size *5

neighbors_t = get_neighbors_time(Seg.analogsignals[0],st,n_samples,n_neighbors)

print_msg("Calculated window time %f based on %d neighbours"%(neighbors_t,n_neighbors))

# Getting longer templates

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

#########################################################################################################
####    get average spikes
#########################################################################################################

n_samples = np.sum(w_samples)

default_templates = True

if not default_templates:
    # Get average shapes
    mode = 'end'
    aligned_spikes = align_spikes(Templates,mode)
    average_spikes = get_averages_from_units(aligned_spikes,units,SpikeInfo,unit_column)

else:
    A = np.load("./templates/template_a.npy")
    B = np.load("./templates/template_b.npy")

    average_spikes = [A,B]

#label spike clusters
amps = [max(av)-min(av) for av in average_spikes]

a_id = np.argmax(amps)
b_id = np.argmin(amps)


titles={a_id:'a',b_id:'b'}
unit_titles={units[a_id]:'a',units[b_id]:'b','-2':'?','-1':'-'}
title_units = {'a':units[a_id],'b':units[b_id],'?':'-2','-':'-1'}

A = average_spikes[a_id]
B = average_spikes[b_id]
# print(titles)
print_msg("Labeled units %s"%unit_titles)

outpath = results_folder / 'template_a.npy'
sp.save(outpath, A)
outpath = results_folder / 'template_b.npy'
sp.save(outpath, B)


# Get combined templates from averages
mode = 'end' # Alignment mode
dt_c = 2 # moving step to sum signals for different combinations
max_len = n_samples # max extension of the combined template

#Get combination from a,b; b,a; a; b; a+b (c)
combined_templates,templates_labels = get_combined_templates([A,B],dt_c,max_len,mode)

#Plot templates
# plot_combined_templates(combined_templates,templates_labels,ncols=5)
# plt.show()

#########################################################################################################
####   Calculate distance from each spike to template
#########################################################################################################
# mode = 'end'
# mode = 'mean'
# mode = 'neighbors'
lim = 120

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

else:
    long_waveforms = get_Templates(data, inds, (w_samples[0],w_samples[1]+max_len)).T
    long_waveforms_align = align_spikes(long_waveforms,mode=mode)
    aligned_templates = np.array(combined_templates)
    # long_waveforms = np.array([align_to(np.concatenate((t,[t[-1]]*max_len)),mode) for t in Templates.T])
    distances=distance_to_average(long_waveforms.T,combined_templates)


#########################################################################################################
####    compare spikes with templates  and reassign
#########################################################################################################

#Compare all templates
colors = get_colors(units)

# # #TODO: fix no need to assign color to -1 and -2...
colors[title_units['a']] = 'g'
colors[title_units['b']] = 'b'
colors['-1'] = 'k'
colors['-2'] = 'k'
print(colors)
# # print(colors)

# t_colors = [colors[unit] for unit in SpikeInfo[unit_column]]
# print()

labeled = []

#New column for result
new_labels = copy.deepcopy(SpikeInfo[unit_column].values)
new_column = 'unit_reassign'
SpikeInfo[new_column] = new_labels

for t_id,t in enumerate(long_waveforms_align):
    #Get this peak and next one
    try:
        peak = SpikeInfo['time'][t_id] 
        next_peak = SpikeInfo['time'][t_id+1]
    except:
        continue

    #Get spike unit and next one
    my_unit = SpikeInfo[unit_column][t_id]
    next_unit = SpikeInfo[unit_column][t_id+1]

    # If its unit -2, label by best match.
    if int(my_unit) < 0:
        # #Reasonable distance to next spike
        w_time = (n_samples//2)/10000

        # if next_peak < peak+w_time: #If spikes are close enough

        # best_match = templates_labels[np.argmin(distances[t_id])]

        #Gets 2 best matches
        best_match_ids = np.argsort(distances[t_id])[:2]

        #TODO: review this restriction. See example ----
        #If best 2 distances are similar --> choose the one with two labels 
        a_error = 0.01
        if abs(distances[t_id][best_match_ids[0]]-distances[t_id][best_match_ids[1]]) < a_error:

            best_match_dis = [len(dis) for dis in np.array(templates_labels)[best_match_ids]]
            best_match = templates_labels[best_match_ids[np.argmax(best_match_dis)]]
            small_diff.append(t_id)
            # print("Distance values so close %.3f %.3f, changing match from %s to %s"%(distances[t_id][best_match_ids[0]],distances[t_id][best_match_ids[1]],templates_labels[np.argmin(distances[t_id])],best_match))

        else:
            best_match = templates_labels[np.argmin(distances[t_id])]


        # print("guess:",best_match,distances[t_id,np.argmin(distances[t_id])])
        # print("current:",curr_match)
        # print(best_match != [unit_titles[my_unit],unit_titles[next_unit]])

        SpikeInfo[new_column][t_id] = title_units[best_match[0]]
        labeled.append(t_id)

print_msg("Number of spikes labeled: %d"%len(labeled))

# for t_id in to_change:
for t_id in labeled:
    peak = SpikeInfo['time'][t_id] 
    t = long_waveforms_align[t_id].T

    zoom = [peak- neighbors_t ,peak+neighbors_t]
    print(t_id)
    fig, axes=plot_compared_fitted_spikes(Seg, 0, Templates[:40,:], SpikeInfo, [unit_column, new_column], zoom=zoom, save=None,colors=colors)
    axes[0].plot([peak,next_peak],[1,1],'.',markersize=5,color='r')
    axes[1].plot([peak,next_peak],[1,1],'.',markersize=5,color='r')
    outpath = plots_folder / ('-2_changed_spike_'+str(t_id)+'_signal' + fig_format)
    plt.savefig(outpath)
    plt.close()

    # title = "spike %d from unit %s"%(t_id,unit_titles[SpikeInfo[unit_column][t_id]])
    # outpath = plots_folder / ('spike_'+str(t_id)+'_templates_grid_all' + fig_format)
    # plot_combined_templates(aligned_templates[t_id],templates_labels,ncols=8,org_spike=t,distances=distances[t_id],title=title,save=outpath)
    # #TODO change when combined templates is general not working
    # #     plot_combined_templates(combined_templates,templates_labels,ncols=5)

    title = "spike %d from unit %s"%(t_id,unit_titles[SpikeInfo[unit_column][t_id]])
    outpath = plots_folder / ('-2_spike_'+str(t_id)+'_templates_grid' + fig_format)
    try:
        plot_combined_templates_bests(aligned_templates[t_id][:lim,:],templates_labels,org_spike=t[:lim],distances=distances[t_id],title=title,save=outpath)
    except:
        plot_combined_templates_bests(aligned_templates[:lim,:],templates_labels,org_spike=t[:lim],distances=distances[t_id],title=title,save=outpath)


