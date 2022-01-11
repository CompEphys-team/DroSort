import matplotlib.pyplot as plt
import gc
import pandas as pd
import configparser

from sssio import * 
from plotters import *
from functions import *
from sys import path
# from superpos_functions import align_spike
from postprocessing_functions import *


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
plots_folder = results_folder / 'plots' / 'post_processing' / 'templates'

os.makedirs(plots_folder, exist_ok=True)

Blk=get_data(results_folder/"result.dill")
SpikeInfo = pd.read_csv(results_folder/"SpikeInfo.csv")

unit_column = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
SpikeInfo = SpikeInfo.astype({unit_column: str})
units = get_units(SpikeInfo, unit_column)
Templates_ini = np.load(results_folder/"Templates_ini.npy")


mpl.rcParams['figure.dpi'] = Config.get('output','fig_dpi')
fig_format = Config.get('output','fig_format')
plotting_changes = Config.getboolean('postprocessing','plot_changes')
complete_grid = Config.getboolean('postprocessing','complete_grid')

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
n_neighbors = 6#TODO add to config file
n_samples = Templates_ini[:,0].size *5

neighbors_t = get_neighbors_time(Seg.analogsignals[0],st, n_samples, n_neighbors)

print_msg("Calculated window time %f based on %d neighbours"%(neighbors_t, n_neighbors))

# Getting longer templates

print_msg(' - getting templates - ')

fs = Blk.segments[0].analogsignals[0].sampling_rate
# n_samples = (wsize * fs).simplified.magnitude.astype('int32')

w_samples = (20, 70)

templates = []
for j, seg in enumerate(Blk.segments):
    data = seg.analogsignals[0].magnitude.flatten()
    inds = (seg.spiketrains[0].times * fs).simplified.magnitude.astype('int32')
    templates.append(get_Templates(data, inds, w_samples))

Templates = sp.concatenate(templates, axis=1)

#########################################################################################################
####    get average spikes
#########################################################################################################

n_samples = np.sum(w_samples)

default_templates = True
mode = 'end'

aligned_spikes = align_spikes(Templates, mode)

if not default_templates:
    # Get average shapes
    average_spikes = get_averages_from_units(aligned_spikes, units, SpikeInfo, unit_column)

    # plot_templates(Templates, SpikeInfo, unit_column)
    # plot_averages(average_spikes, SpikeInfo, unit_column)
    # plt.show()


    # #label spike clusters
    # amps = [max(av)-min(av) for av in average_spikes]

    # a_id = np.argmax(amps)
    # b_id = np.argmin(amps)


    # titles={a_id:'a',b_id:'b'}
    # unit_titles={units[a_id]:'a',units[b_id]:'b','-2':'?','-1':'-'}
    # title_units = {'a':units[a_id],'b':units[b_id],'?':'-2','-':'-1'}

    # A = average_spikes[a_id]
    # B = average_spikes[b_id]
    # # print(titles)
    # print_msg("Labeled units %s"%unit_titles)
    # # plot_averages(average_spikes, SpikeInfo, unit_column, title=titles)
    # # plt.show()


else:
    A = np.load("./templates/template_a.npy")
    B = np.load("./templates/template_b.npy")

    average_spikes = [A, B]

# label spike clusters
amps = [max(av) - min(av) for av in average_spikes]

a_id = np.argmax(amps)
b_id = np.argmin(amps)


titles = {a_id: 'a', b_id: 'b'}
unit_titles = {units[a_id]: 'a', units[b_id]: 'b', '-2': '?', '-1': '-'}
title_units = {'a': units[a_id], 'b': units[b_id], '?': '-2', '-': '-1'}

A = average_spikes[a_id]
B = average_spikes[b_id]
# print(titles)
print_msg("Labeled units %s" % unit_titles)

outpath = results_folder / 'template_a.npy'
sp.save(outpath, A)
outpath = results_folder / 'template_b.npy'
sp.save(outpath, B)


# Get combined templates from averages
mode = 'end'  # Alignment mode
dt_c = 2  # moving step to sum signals for different combinations
max_len = n_samples  # max extension of the combined template

# Get combination from a, b; b, a; a; b; a+b (c)
combined_templates, templates_labels = get_combined_templates([A, B],dt_c, max_len, mode)

ncols = 10  # TODO add to config file
# Plot templates

outpath = plots_folder / ('general_combined_templates_grid' + fig_format)
plot_combined_templates(np.array(combined_templates),templates_labels, ncols=ncols, save=outpath)
# plt.show()

#########################################################################################################
####   Calculate distance from each spike to template
#########################################################################################################

#mode == mean, neighbors or alignment mode 'end', 'ini', 'peak','min'
mode = Config.get('postprocessing', 'mode_templates')
lim = Config.getint('postprocessing', 'lim_templates')

# Get distances
if mode == 'neighbors' or mode == 'mean':
    long_waveforms = get_Templates(data, inds, (w_samples[0], w_samples[1] + max_len)).T

    aligned_templates = np.zeros((long_waveforms.shape[0], len(combined_templates), long_waveforms.shape[1]))[:, :, :lim]
    long_waveforms_align = np.zeros(long_waveforms.shape)[:, :lim]
    all_combined_templates = np.zeros((long_waveforms.shape[0], len(combined_templates), long_waveforms.shape[1]))

    D_pw = sp.zeros((long_waveforms_align.shape[0], aligned_templates.shape[1]))

    #TODO: simplify code, put together mean alignment for neighbors and mean
    if mode == 'mean':
        for t_i, t in enumerate(long_waveforms[:, :lim]):
            for ct_i, ct in enumerate(combined_templates[:, :lim]):
                d_mean = (np.mean(ct) - np.mean(t))**2

                ct_a = align_to(ct, d_mean)
                t_a = align_to(t, d_mean)

                long_waveforms_align[t_i] = t_a
                aligned_templates[t_i, ct_i] = ct_a

            D_pw[t_i, :] = metrics.pairwise.euclidean_distances(aligned_templates[t_i],t.reshape(1,-1)).reshape(-1)

        distances = D_pw

    elif mode == 'neighbors':
        short_size = 90
        short_aligned_spikes = aligned_spikes[:short_size,:]

        for t_i, t in enumerate(long_waveforms):
            spike_id = SpikeInfo['id'][t_i]
            # get neighbors
            # get ids from time reference 
            neighbors_ids = get_spikes_ids(t_i, neighbors_t, SpikeInfo, st)[0]

            # get average template
            # get spikes from short template, not long version by changing short_size value
            average_spikes = get_averages_from_units(short_aligned_spikes, units, SpikeInfo.loc[neighbors_ids],unit_column, verbose=False)

            # get combined templates for the new averages
            all_combined_templates[t_i],templates_labels = get_combined_templates(average_spikes, dt_c, max_len, mode='end')

            # mode = 'peak'
            # align both by mean
            for ct_i, ct in enumerate(all_combined_templates[t_i][:lim]):
                d_mean = (np.mean(ct) - np.mean(t))**2
                # print(d_mean)
                ct_a = align_to(ct, d_mean)
                t_a = align_to(t, d_mean)
                #TODO: neighbors with different alignment?
                # ct_a = align_to(ct,'peak')
                # t_a = align_to(t,'peak')
                long_waveforms_align[t_i] = t_a[:lim]
                aligned_templates[t_i, ct_i] = ct_a[:lim]

            # get distances
            D_pw[t_i, :] = metrics.pairwise.euclidean_distances(aligned_templates[t_i],long_waveforms_align[t_i].reshape(1,-1)).reshape(-1)

        distances = D_pw
else:
    long_waveforms = get_Templates(data, inds, (w_samples[0], w_samples[1] + max_len)).T
    long_waveforms_align = align_spikes(long_waveforms, mode=mode)
    aligned_templates = np.array(combined_templates)
    distances = distance_to_average(long_waveforms.T, combined_templates)


#########################################################################################################
####    compare spikes with templates  and reassign
#########################################################################################################

# Compare all templates
colors = get_colors(units)

# # #TODO: fix no need to assign color to -1 and -2...
# colors[title_units['a']] = 'g'
# colors[title_units['b']] = 'b'
# colors['-1'] = 'k'
# colors['-2'] = 'k'

c_spikes = []
non_spikes = []
to_change = []
small_diff = []

# New column for result
new_labels = copy.deepcopy(SpikeInfo[unit_column].values)
new_column = 'unit_templates'
SpikeInfo[new_column] = new_labels

for t_id, t in enumerate(long_waveforms_align):
    # Get this peak and next one
    try:
        peak = SpikeInfo['time'][t_id]
        next_peak = SpikeInfo['time'][t_id + 1]
    except:
        continue

    if t_id in non_spikes:
        continue

    # Get spike unit and next one
    my_unit = SpikeInfo[unit_column][t_id]
    next_unit = SpikeInfo[unit_column][t_id + 1]

    # Reasonable distance to next spike
    w_time = (n_samples//2)/10000

    # Gets 2 best matches
    best_match_ids = np.argsort(distances[t_id])[:2]

    # TODO: review this restriction. See example ----
    # If best 2 distances are similar --> choose the one with two labels 
    a_error = 0.03
    if abs(distances[t_id][best_match_ids[0]] - distances[t_id][best_match_ids[1]]) < a_error and templates_labels[best_match_ids[0]] != templates_labels[best_match_ids[1]]:

        best_match_dis = [len(dis) for dis in np.array(templates_labels)[best_match_ids]]
        best_match = templates_labels[best_match_ids[np.argmax(best_match_dis)]]
        small_diff.append(t_id)
        # print("Distance values so close %.3f %.3f, changing match from %s to %s"%(distances[t_id][best_match_ids[0]],distances[t_id][best_match_ids[1]],templates_labels[np.argmin(distances[t_id])],best_match))

    else:
        best_match = templates_labels[np.argmin(distances[t_id])]

    # print("guess:",best_match, distances[t_id, np.argmin(distances[t_id])])
    # print("current:",curr_match)
    # print(best_match != [unit_titles[my_unit],unit_titles[next_unit]])

    curr_match = [unit_titles[my_unit], unit_titles[next_unit]]


    diff = (to_points(next_peak, fs)+w_samples[1]) - (to_points(peak, fs)-w_samples[0])
    # print(diff, n_samples)

    # TODO: mixed restriction "lim" parameter and this diff!!!

    # If spikes are not close enough, get only best match.
    if diff > n_samples * 2 - 20:

        best_match = best_match[0]
        curr_match = best_match[0]

    # guess different to assignation
    if best_match != curr_match:
        # print("spike %d from unit %s"%(t_id, unit_titles[SpikeInfo[unit_column][t_id]]))

        # Skip b correct matches (saving time in plot)
        if best_match == ['b'] and curr_match == ['b','b']:
            # print("Skiping b match",best_match)
            continue

        if best_match == ['a'] and curr_match == ['a','b'] and next_peak < peak+w_time:  # If spikes are close enough is an a + rebound
        # if best_match == ['a'] and curr_match == ['a','b']:
            non_spikes.append(t_id + 1)

        elif best_match[0] == 'c':
            c_spikes.append(t_id)
            non_spikes.append(t_id + 1)

        else:  # general case change current unit and next one by guess
            SpikeInfo[new_column][t_id] = title_units[best_match[0]]
            to_change.append(t_id)
            try:
                SpikeInfo[new_column][t_id + 1] = title_units[best_match[1]]
                to_change.append(t_id + 1)
            except:  # when guess is only for 1 spike
                pass

print_msg("Number of spikes relabeled: %d" % len(to_change))
print_msg("Number of spikes 'non_spikes': %d" % len(non_spikes))
print_msg("Number of spikes 'large' combined: %d" % len(c_spikes))

# Unassign rebounds
SpikeInfo[new_column].iloc[non_spikes] = '-1'
# Warning: the spike will still be in Templates

# # Modify c spike

# SpikeInfo[new_column].iloc[c_spikes] = title_units['a']
# # 1. Add spike in SpikeInfo
# id_l = max(SpikeInfo['id'])  # get last id
# # index = len(SpikeInfo[new_column])  # get last index
# # copy c spike rows
# print(SpikeInfo[new_column].value_counts())
# SpikeInfo = pd.concat([SpikeInfo, SpikeInfo.iloc[c_spikes]])

# # 2. Add spike in Template
# # Templates.reshape(Templates.shape[0]+len(c_spikes),Templates.shape[1])
# print(Templates.shape)
# print(SpikeInfo[new_column].value_counts())

# print_msg("Adding sum spikes")
# print(c_spikes)
# # Sort spike and id
# for i, s in enumerate(c_spikes):
#     new_id = i + id_l  # get new id in the end of Spike Info
#     # get original id, that is the same
#     org_id = SpikeInfo['id'].iloc[new_id]  # get index

#     # Templates[:,new_id] = Templates[:,org_id]
#     Templates = np.append(Templates, Templates[:, org_id, np.newaxis], axis=1)

#     SpikeInfo[new_column].iloc[new_id] = title_units['b']
#     SpikeInfo['id'].iloc[new_id] = new_id

# # print(SpikeInfo.keys())
# # print(SpikeInfo[new_column].value_counts())

# SpikeInfo = SpikeInfo.sort_values(by='time')

# print(SpikeInfo[new_column].value_counts())

# # TODO: fix Blk spiketrain, fails in plotting waveforms
# c_times = SpikeInfo.iloc[c_spikes]['time'].values  # use id ¿?
# c_peaks = np.max(Templates[:, c_spikes], axis=0)

# print("New column")
# print(c_peaks)
# print(c_times)
# print(len(Blk.segments[0].spiketrains))

# Blk = add_spikes_to_SpikeTrain(Blk, c_times, c_peaks)
# # TODO Blk 3 spiketrains ¿?¿?¿?
# print(len(Blk.segments))
# print(len(Blk.segments[0].spiketrains))
# print(Blk.segments[0].spiketrains[0].times.shape)
# print(Blk.segments[0].spiketrains[1].times.shape)
# print(Blk.segments[0].spiketrains[2].times.shape)
# print(Blk.segments[0].spiketrains[0].waveforms.shape)

# print_msg("Saving SpikeInfo, Blk and Spikes into disk")

# units = get_units(SpikeInfo, new_column)
# Blk = populate_block(Blk, SpikeInfo, new_column, units)
# # save_all(results_folder, Config, SpikeInfo, Blk, units)


outpath = results_folder / 'Templates_final.npy'
sp.save(outpath, Templates[:40, :])

# plot all sorted spikes
for j, Seg in enumerate(Blk.segments):
    seg_name = Path(Seg.annotations['filename']).stem
    outpath = plots_folder / (seg_name + '_overview' + fig_format)
    plot_segment(Seg, units, save=outpath)

# plot all sorted spikes
max_window = 0.3  # AG: TODO add to config file
plot_fitted_spikes_complete(Blk, Templates[:40, :], SpikeInfo, unit_column,
                            max_window, plots_folder, fig_format, wsize=40, extension='_templates')

print_msg("general plotting done")

if plotting_changes:

    # list all changes and labels for output
    labels = ['non_spike_'] * len(non_spikes) + ['changed_spike_'] * len(to_change)\
        + ['sum_spikes_'] * len(c_spikes) + ['small_diff_spike_'] * len(small_diff)
    all_changes = non_spikes + to_change + c_spikes + small_diff

    print_msg("plotting changes")

    # Plot every change
    for t_id, label in zip(all_changes, labels):
        peak = SpikeInfo['time'][t_id]
        t = long_waveforms_align[t_id - 1].T

        title = "spike %d from unit %s" % (t_id, unit_titles[SpikeInfo[unit_column].iloc[t_id]])

        zoom = [peak - neighbors_t, peak + neighbors_t]
        # print(t_id)

        fig, axes = plot_compared_fitted_spikes(Seg, 0, Templates[:40, :], SpikeInfo, [unit_column, new_column], zoom=zoom, save=None, title=title)
        axes[0].plot([peak, next_peak], [1, 1], '.', markersize=5, color='r')
        axes[1].plot([peak, next_peak], [1, 1], '.', markersize=5, color='r')
        outpath = plots_folder / (label + str(t_id) + '_signal' + fig_format)
        plt.savefig(outpath)
        # WARNING: do not add plt.close; figure clears by definition
        #         (arg: num=1, clear=True) adding plt.close leaks memory

        if label == 'non_spike_':  # a non-spike is removed at the previous peak
            id_ = t_id - 1
        else:
            id_ = t_id

        try:
            temp = aligned_templates[id_]
        except:
            temp = aligned_templates

        if complete_grid:
            outpath = plots_folder / (label + str(t_id) + '_templates_grid_all' + fig_format)
            plot_combined_templates(temp, templates_labels, ncols=ncols, org_spike=t, distances=distances[id_], title=title, save=outpath)

        outpath = plots_folder / (label + str(t_id) + '_templates_grid' + fig_format)

        
        plot_combined_templates_bests(temp, templates_labels, org_spike=t, distances=distances[id_],title=title, save=outpath)

        # WARNING: do not add plt.close; figure clears by definition
        #         (arg: num=1, clear=True) adding plt.close leaks memory


print_msg("all done - quitting")
