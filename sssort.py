# sys
import sys
import os
import copy
import dill
import shutil
import configparser
from pathlib import Path
from tqdm import tqdm

# sci
import scipy as sp
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ephys
import neo
import elephant as ele

# own
from functions import *
from plotters import *
import sssio

# banner
if os.name == "posix":
    tp.banner("This is SSSort v1.0.0", 78)
    tp.banner("author: Georg Raiser - grg2rsr@gmail.com", 78)
else:
    print("This is SSSort v1.0.0")
    print("author: Georg Raiser - grg2rsr@gmail.com")

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

"""
 
 #### ##    ## #### 
  ##  ###   ##  ##  
  ##  ####  ##  ##  
  ##  ## ## ##  ##  
  ##  ##  ####  ##  
  ##  ##   ###  ##  
 #### ##    ## #### 
 
"""

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
plots_folder = results_folder / 'plots'
os.makedirs(plots_folder, exist_ok=True)
os.chdir(config_path.parent / exp_name)

#TODO: add question!!
os.system("rm %s/*"%plots_folder)

# copy config
shutil.copyfile(config_path, config_path.parent / exp_name / config_path.name)

# read data
Blk = sssio.get_data(data_path)
Blk.name = exp_name
print_msg('data read from %s' % data_path)

# plotting
mpl.rcParams['figure.dpi'] = Config.get('output','fig_dpi')
fig_format = Config.get('output','fig_format')

# try:
#     ini = Config.getint('preprocessing','ini') // 1000
#     end = Config.getint('preprocessing','end') // 1000
# except configparser.NoOptionError as no_option:
#     ini = 0
#     end = -1
# except:
#     exit()

"""
 
 ########  ########  ######## ########  ########   #######   ######  ########  ######   ######  
 ##     ## ##     ## ##       ##     ## ##     ## ##     ## ##    ## ##       ##    ## ##    ## 
 ##     ## ##     ## ##       ##     ## ##     ## ##     ## ##       ##       ##       ##       
 ########  ########  ######   ########  ########  ##     ## ##       ######    ######   ######  
 ##        ##   ##   ##       ##        ##   ##   ##     ## ##       ##             ##       ## 
 ##        ##    ##  ##       ##        ##    ##  ##     ## ##    ## ##       ##    ## ##    ## 
 ##        ##     ## ######## ##        ##     ##  #######   ######  ########  ######   ######  
 
"""

print_msg(' - preprocessing - ')

for seg in Blk.segments:
    seg.analogsignals[0].annotate(kind='original')
    # seg.analogsignals[0] = seg.analogsignals[0][ini:end]

# highpass filter
freq = Config.getfloat('preprocessing','highpass_freq')
print_msg("highpass filtering data at %.2f Hz" % freq)
for i, seg in enumerate(Blk.segments):
    seg.analogsignals[0] = ele.signal_processing.butter(seg.analogsignals[0], highpass_freq=freq)
    if 'filename' not in seg.annotations:
        seg.annotations['filename']= 'segment_'+str(i)

# invert if peaks are negative
if Config.get('preprocessing','peak_mode') == 'negative':
    print_msg("inverting signal")
    for seg in Blk.segments:
        seg.analogsignals[0] *= -1

if Config.getboolean('preprocessing','z_trials'):
    print_msg("z-scoring analogsignals for each trial")
    for seg in Blk.segments:
        seg.analogsignals = [ele.signal_processing.zscore(seg.analogsignals)]

"""
 
  ######  ########  #### ##    ## ########    ########  ######## ######## ########  ######  ######## 
 ##    ## ##     ##  ##  ##   ##  ##          ##     ## ##          ##    ##       ##    ##    ##    
 ##       ##     ##  ##  ##  ##   ##          ##     ## ##          ##    ##       ##          ##    
  ######  ########   ##  #####    ######      ##     ## ######      ##    ######   ##          ##    
       ## ##         ##  ##  ##   ##          ##     ## ##          ##    ##       ##          ##    
 ##    ## ##         ##  ##   ##  ##          ##     ## ##          ##    ##       ##    ##    ##    
  ######  ##        #### ##    ## ########    ########  ########    ##    ########  ######     ##    

"""

print_msg('- spike detect - ')

# detecting all spikes by MAD thresholding
mad_thresh = Config.getfloat('spike detect', 'mad_thresh')
wsize = Config.getfloat('spike detect', 'wsize') * pq.ms


bad_segments = []
for i, seg in enumerate(Blk.segments):
    AnalogSignal, = select_by_dict(seg.analogsignals, kind='original')
    bounds = [MAD(AnalogSignal)*mad_thresh, sp.inf] * AnalogSignal.units
    bounds_neg = [MAD(AnalogSignal)*(mad_thresh-2), sp.inf] * AnalogSignal.units #TODO hardcode mad_thresh lower
    # bounds_neg = [-sp.inf,-MAD(AnalogSignal)*(mad_thresh-2)] * AnalogSignal.units #TODO hardcode mad_thresh lower


    fs = Blk.segments[0].analogsignals[0].sampling_rate
    n_samples = (wsize * fs).simplified.magnitude.astype('int32')

    if Config.get('preprocessing','peak_mode') == 'double':
        # st = spike_detect(AnalogSignal, [0,sp.inf])
        st = double_spike_detect(AnalogSignal, bounds, bounds_neg,wsize=n_samples)
    else:
        st = spike_detect(AnalogSignal, bounds,wsize=n_samples)

    if st.times.shape[0] == 0:
        stim_name = Path(seg.annotations['filename']).stem
        print_msg("no spikes found for segment %i:%s" % (i,stim_name))
        bad_segments.append(i)
    st.annotate(kind='all_spikes')

    # remove border spikes
    wsize = Config.getfloat('spike detect', 'wsize') * pq.ms
    st_cut = st.time_slice(st.t_start + wsize/2, st.t_stop - wsize/2)
    st_cut.t_start = st.t_start


    st_cut = reject_non_spikes(AnalogSignal,st_cut,n_samples,verbose=True,plot=False)

    seg.spiketrains.append(st_cut)

n_spikes = sp.sum([seg.spiketrains[0].shape[0] for seg in Blk.segments])
print_msg("total number of spikes found: %s" % n_spikes)


#Plot detected spikes
for i,seg in enumerate(Blk.segments):
    namepath = plots_folder / ("first_spike_detection_%d"%i)
    plot_spike_events(seg,thres=MAD(AnalogSignal)*mad_thresh,save=namepath,save_format=fig_format,show=False)

print_msg("detected spikes plotted")

#Detect bad segments based on norm probability distribution

if Config.getboolean('preprocessing', 'sd_reject'):
    stim_onset = Config.getfloat('preprocessing', 'stim_onset') * pq.s
    alpha = Config.getfloat('preprocessing', 'sd_reject_alpha')
    sd_rej_fac = stats.distributions.norm.isf(alpha) # hardcoded

    Peaks = []
    for i, Seg in enumerate(Blk.segments):
        peaks = get_all_peaks([Seg], lowpass_freq=1*pq.kHz, t_max=stim_onset)
        Peaks.append(peaks)

    mus = sp.array([sp.average(peaks) for peaks in Peaks])
    sigs = sp.array([sp.std(peaks) for peaks in Peaks])
    grand_mu = sp.average([sp.average(peaks) for peaks in Peaks])
    grand_sig = sp.std([sp.average(peaks) for peaks in Peaks])

    bad_trials = sp.logical_or(mus < (grand_mu - sd_rej_fac * grand_sig), mus > (grand_mu + sd_rej_fac * grand_sig))
    [bad_segments.append(i) for i in sp.where(bad_trials)[0]]
    print_msg("rejecting %i out of %i trials" % (sum(bad_trials), bad_trials.shape[0]))

if len(bad_segments) > 0:
    good_segments = []
    for i, seg in enumerate(Blk.segments):
        if i not in bad_segments:
            good_segments.append(seg)
        else:
            stim_name = Path(Blk.segments[i].annotations['filename']).stem
            print_msg("rejecting: %i:%s" % (i,stim_name))

    Blk_all = Blk.segments
    Blk.segments = good_segments

    n_spikes = sp.sum([seg.spiketrains[0].shape[0] for seg in Blk.segments])
    print_msg("total number of spikes left: %s" % n_spikes)


    #Plot detected spikes after rejection
    for i,(seg_all,seg_good) in enumerate(zip(Blk.segments,Blk_all)):
        namepath = plots_folder / ("first_spike_detection_%d"%i)
        plot_compared_spike_events(seg_all,seg_good,thres=MAD(AnalogSignal)*mad_thresh,save=namepath,save_format=fig_format,show=True)

    print_msg("detected spikes plotted")


"""
 
 ######## ######## ##     ## ########  ##          ###    ######## ########  ######  
    ##    ##       ###   ### ##     ## ##         ## ##      ##    ##       ##    ## 
    ##    ##       #### #### ##     ## ##        ##   ##     ##    ##       ##       
    ##    ######   ## ### ## ########  ##       ##     ##    ##    ######    ######  
    ##    ##       ##     ## ##        ##       #########    ##    ##             ## 
    ##    ##       ##     ## ##        ##       ##     ##    ##    ##       ##    ## 
    ##    ######## ##     ## ##        ######## ##     ##    ##    ########  ######  
 
"""

print_msg(' - getting templates - ')

fs = Blk.segments[0].analogsignals[0].sampling_rate
n_samples = (wsize * fs).simplified.magnitude.astype('int32')

templates = []
for j, seg in enumerate(Blk.segments):
    data = seg.analogsignals[0].magnitude.flatten()
    inds = (seg.spiketrains[0].times * fs).simplified.magnitude.astype('int32')
    templates.append(get_Templates(data, inds, n_samples))

Templates = sp.concatenate(templates,axis=1)

# templates to disk
outpath = results_folder / 'Templates.npy'
sp.save(outpath, Templates)
print_msg("saving Templates to %s" % outpath)

zoom = sp.array(Config.get('output','zoom').split(','),dtype='float32') / 1000
for j,Seg in enumerate(Blk.segments):
    outpath = plots_folder / ("templates_in_signal_init"+fig_format)
    plot_templates_on_trace(Seg, j, Templates, save=outpath,wsize=n_samples,zoom=zoom)



"""
 
  ######  ##       ##     ##  ######  ######## ######## ########  
 ##    ## ##       ##     ## ##    ##    ##    ##       ##     ## 
 ##       ##       ##     ## ##          ##    ##       ##     ## 
 ##       ##       ##     ##  ######     ##    ######   ########  
 ##       ##       ##     ##       ##    ##    ##       ##   ##   
 ##    ## ##       ##     ## ##    ##    ##    ##       ##    ##  
  ######  ########  #######   ######     ##    ######## ##     ## 
 
"""

n_clusters_init = Config.getint('spike sort','init_clusters')
print_msg("initial kmeans with %i clusters" % n_clusters_init)
pca = PCA(n_components=5) # FIXME HARDCODED PARAMETER
X = pca.fit_transform(Templates.T)
kmeans_labels = KMeans(n_clusters=n_clusters_init).fit_predict(X)
spike_labels = kmeans_labels.astype('U')

"""
 
  ######  ########  #### ##    ## ######## #### ##    ## ########  #######  
 ##    ## ##     ##  ##  ##   ##  ##        ##  ###   ## ##       ##     ## 
 ##       ##     ##  ##  ##  ##   ##        ##  ####  ## ##       ##     ## 
  ######  ########   ##  #####    ######    ##  ## ## ## ######   ##     ## 
       ## ##         ##  ##  ##   ##        ##  ##  #### ##       ##     ## 
 ##    ## ##         ##  ##   ##  ##        ##  ##   ### ##       ##     ## 
  ######  ##        #### ##    ## ######## #### ##    ## ##        #######  
 
"""
#  make a SpikeInfo dataframe
SpikeInfo = pd.DataFrame()

# count spikes
n_spikes = Templates.shape[1]
SpikeInfo['id'] = sp.arange(n_spikes,dtype='int32')

# get all spike times
spike_times = sp.concatenate([seg.spiketrains[0].times.magnitude for seg in Blk.segments])
SpikeInfo['time'] = spike_times

# get segment labels
segment_labels = []
for i, seg in enumerate(Blk.segments):
    segment_labels.append(seg.spiketrains[0].shape[0] * [i])
segment_labels = sp.concatenate(segment_labels)
SpikeInfo['segment'] = segment_labels

# get all labels
SpikeInfo['unit'] = spike_labels #AG: Unit column has cluster id.

# get clean templates
n_neighbors = Config.getint('spike model','template_reject')
reject_spikes(Templates, SpikeInfo, 'unit', n_neighbors, verbose=True)

# unassign spikes if unit has too little good spikes
SpikeInfo = unassign_spikes(SpikeInfo, 'unit',min_good=15)

outpath = plots_folder / ("templates_init" + fig_format)
plot_templates(Templates, SpikeInfo, N=100, save=outpath)


"""
 
 #### ##    ## #### ######## 
  ##  ###   ##  ##     ##    
  ##  ####  ##  ##     ##    
  ##  ## ## ##  ##     ##    
  ##  ##  ####  ##     ##    
  ##  ##   ###  ##     ##    
 #### ##    ## ####    ##    
 
"""
# first ini run
print_msg('- initializing algorithm: calculating all initial firing rates')

# rate est
kernel_slow = Config.getfloat('kernels','sigma_slow')
kernel_fast = Config.getfloat('kernels','sigma_fast')
calc_update_frates(Blk.segments, SpikeInfo, 'unit', kernel_fast, kernel_slow)

# model
n_model_comp = Config.getint('spike model','n_model_comp')
Models = train_Models(SpikeInfo, 'unit', Templates, n_comp=n_model_comp, verbose=True)
outpath = plots_folder / ("Models_ini" + fig_format)
plot_Models(Models, save=outpath)

zoom = sp.array(Config.get('output','zoom').split(','),dtype='float32') / 1000


seg_name = Path(Seg.annotations['filename']).stem
units = get_units(SpikeInfo, 'unit')


Blk = populate_block(Blk,SpikeInfo,'unit',units)

Seg = Blk.segments[0]
outpath = plots_folder / (seg_name + '_fitted_spikes_init' + fig_format)
plot_fitted_spikes(Seg, j, Models, SpikeInfo, 'unit', zoom=zoom, save=outpath,wsize=n_samples)

# #FIX: Template seems like model?¿?¿?¿?
# max_window = 0.3 #AG: TODO add to config file
# plot_fitted_spikes_complete(Blk, Templates, SpikeInfo, 'unit', max_window, plots_folder, fig_format,wsize=n_samples,extension='_templates')


# max_window = 0.3 #AG: TODO add to config file

# for j, Seg in enumerate(Blk.segments):
#     seg_name = Path(Seg.annotations['filename']).stem

#     asig = Seg.analogsignals[0]
#     max_window = int(max_window*asig.sampling_rate) #FIX conversion from secs to points
#     n_plots = asig.shape[0]//max_window

#     for n_plot in range(0,n_plots):
#         outpath = plots_folder / (seg_name + '_templates_%s_%d'%(max_window,n_plot) + fig_format)
#         ini = n_plot*max_window + max_window
#         end = ini + max_window
#         end = min(end, Seg.analogsignals[0].shape[0])
#         zoom = [ini,end]/asig.sampling_rate

#         plot_templates_on_trace(Seg, j, Templates, save=outpath,wsize=n_samples,zoom=zoom)

"""
 
 ########  ##     ## ##    ## 
 ##     ## ##     ## ###   ## 
 ##     ## ##     ## ####  ## 
 ########  ##     ## ## ## ## 
 ##   ##   ##     ## ##  #### 
 ##    ##  ##     ## ##   ### 
 ##     ##  #######  ##    ## 
 
"""

# reset
SpikeInfo['unit_0'] = SpikeInfo['unit'] # the init

n_final_clusters = Config.getint('spike sort','n_final_clusters')
rm_smaller_cluster = Config.getboolean('posprocessing','rm_smaller_cluster')
it_merge = Config.getint('spike sort','it_merge')
org_it_merge = it_merge
first_merge = Config.getint('spike sort','first_merge')
clust_alpha = Config.getfloat('spike sort','clust_alpha')
units = get_units(SpikeInfo, 'unit_0')
n_units = len(units)
penalty = Config.getfloat('spike sort','penalty')
sorting_noise = Config.getfloat('spike sort','f_noise')
ScoresSum = []
AICs = []

spike_ids = SpikeInfo['id'].values

it_no_merge = Config.getint('spike sort','it_no_merge')

it =1
not_merge =0

change_cluster=Config.getint('spike sort','cluster_limit_train')
last = False
while n_units >= n_final_clusters and not last:
    if n_units == n_final_clusters:
        last = True

    # unit columns
    prev_unit_col = 'unit_%i' % (it-1)
    this_unit_col = 'unit_%i' % it
    score = Rss
    
    Scores_old, units = Score_spikes(Templates, SpikeInfo, prev_unit_col, Models, score_metric=score, penalty=penalty)

    # update rates
    calc_update_frates(Blk.segments, SpikeInfo, prev_unit_col, kernel_fast, kernel_slow)

    # train models with labels from last iteration
    Models = train_Models(SpikeInfo, prev_unit_col, Templates, verbose=False, n_comp=n_model_comp)
    outpath = plots_folder / ("Models_%s%s" % (prev_unit_col, fig_format))
    plot_Models(Models, save=outpath)
    if n_units > change_cluster:
        # Score spikes with models
        score = Rss #change to double_score or amplitude_score for other model scoring
        Scores, units = Score_spikes(Templates, SpikeInfo, prev_unit_col, Models, score_metric=score, penalty=penalty)

        # assign new labels
        min_ix = sp.argmin(Scores, axis=1)
        new_labels = sp.array([units[i] for i in min_ix],dtype='object')

    else: #stop clusters changes and force merging
        it_merge = 1
        clust_alpha = 10
        new_labels = sp.array(SpikeInfo[prev_unit_col])

    SpikeInfo[this_unit_col] = new_labels

    n_changes = np.sum(~(SpikeInfo[prev_unit_col]==SpikeInfo[this_unit_col]))
    print("Changes by scoring: %d "%n_changes)
    if it_merge > 1 and n_changes < 3:
        it_merge = 1
        clust_alpha +=0.1
    else:
        it_merge = org_it_merge

    SpikeInfo = unassign_spikes(SpikeInfo, this_unit_col)
    reject_spikes(Templates, SpikeInfo, this_unit_col,verbose=False)

    # randomly unassign a fraction of spikes
    # if it != its-1: # the last
    N = int(n_spikes * sorting_noise)
    SpikeInfo.loc[SpikeInfo.sample(N).index,this_unit_col] = '-1'
    
    # plot templates
    outpath = plots_folder / ("Templates_%s%s" % (this_unit_col, fig_format))
    plot_templates(Templates, SpikeInfo, this_unit_col, save=outpath)

    # every n iterations, merge
    if (it > first_merge) and (it % it_merge) == 0 and not last:
        print_msg("check for merges ... ")
        Avgs, Sds = calculate_pairwise_distances(Templates, SpikeInfo, this_unit_col)
        merge = best_merge(Avgs, Sds, units, clust_alpha)

        if len(merge) > 0:
            print_msg("########merging: " + ' '.join(merge))
            ix = SpikeInfo.groupby(this_unit_col).get_group(merge[1])['id']
            SpikeInfo.loc[ix, this_unit_col] = merge[0]
            
            #reset merging parameters
            not_merge =0
            it_merge = Config.getint('spike sort','it_merge')
            it_no_merge = Config.getint('spike sort','it_no_merge')
        else:
            not_merge +=1

    #Increase merge probability after n failed merges
    if not_merge > it_no_merge:
        clust_alpha +=0.1

        it_merge = max(it_merge-1,1)
        it_no_merge = max(it_no_merge-1,1)

        print_msg("%d failed merges. New alpha value: %f"%(not_merge,clust_alpha))
        not_merge = 0

    # Model eval

    n_changes,Rss_sum,ScoresSum,units,AICs,n_units = eval_model(SpikeInfo,this_unit_col,prev_unit_col,Scores,Templates,ScoresSum,AICs)

    try:
        zoom = sp.array(Config.get('output','zoom').split(','),dtype='float32') / 1000
        

        # for j, Seg in enumerate(Blk.segments):
        seg_name = Path(Seg.annotations['filename']).stem

        # populate_block(Blk,SpikeInfo,prev_unit_col,units)
        Blk = populate_block(Blk,SpikeInfo,this_unit_col,units)

        Seg = Blk.segments[0]

        outpath = plots_folder / (seg_name + '_fitted_spikes_%d'%(it) + fig_format)
        plot_fitted_spikes(Seg, j, Models, SpikeInfo, this_unit_col, zoom=zoom, save=outpath,wsize=n_samples)
    except Exception as ex:
        print(ex.args)
        pass

    try:
        zoom = sp.array(Config.get('output','zoom').split(','),dtype='float32') / 1000
        

        # for j, Seg in enumerate(Blk.segments):
        seg_name = Path(Seg.annotations['filename']).stem

        # populate_block(Blk,SpikeInfo,prev_unit_col,units)
        Blk = populate_block(Blk,SpikeInfo,this_unit_col,units)

        Seg = Blk.segments[0]

        outpath = plots_folder / (seg_name + '_fitted_spikes_%d'%(it) + fig_format)
        plot_fitted_spikes(Seg, j, Models, SpikeInfo, this_unit_col, zoom=zoom, save=outpath,wsize=n_samples)
    except Exception as ex:
        print(ex.args)
        pass


    # print iteration info
    print_msg("It:%i - Rss sum: %.3e - # reassigned spikes: %s / %d" % (it, Rss_sum, n_changes,len(spike_labels)))
    print_msg("Number of clusters after iteration: %d"%len(units))

    it +=1


#######
print_msg("algorithm run is done")

last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]

unit_column = last_unit_col

# #Plot when there's only one trial
# try:
#     get_units(SpikeInfo,unit_column)
# except:
#     SpikeInfo[unit_column] = SpikeInfo['unit']
#     pass

reassigned_amplitude = Config.getboolean('posprocessing','reassign_amplitude')

if reassigned_amplitude:
    units = get_units(SpikeInfo,unit_column)
    amplitudes = get_units_amplitudes(Templates,SpikeInfo,unit_column,lim=10)

    spike_ids = SpikeInfo['id'].values    
    dict_units = {u:i for i,u in enumerate(units)}

    new_labels = copy.deepcopy(SpikeInfo[unit_column].values)

    for i, (spike_id,org_label) in enumerate(zip(spike_ids,SpikeInfo[unit_column])):
        if org_label == '-1':
            continue
        spike = Templates[:, spike_id].T
        ampl = max(spike)-min(spike)

        new_label = units[(dict_units[org_label]+1)%2]

        sur_ampl = get_neighbours_amplitude(Templates,SpikeInfo,unit_column,org_label,idx=spike_id,n=3)
        # if st.times[spike_id] > 10.55 and st.times[spike_id] < 10.65:
            # print(ampl,sur_ampl,amplitudes,dict_units[org_label],st.times[spike_id])

        if ampl > sur_ampl + 0.15 and amplitudes[dict_units[new_label]] > amplitudes[dict_units[org_label]]:
            # print(ampl,sur_ampl,amplitudes,dict_units[org_label],st.times[spike_id])
            new_labels[i] = new_label

    print_msg("Num of final changes %d"%np.sum(~(SpikeInfo[unit_column]==new_labels).values))

SpikeInfo[unit_column] = new_labels

# plot templates and models for last column
outpath = plots_folder / ("Templates_%s%s" % (unit_column,fig_format))
plot_templates(Templates, SpikeInfo, unit_column, save=outpath)
outpath = plots_folder / ("Models_%s%s" % (unit_column,fig_format))
plot_Models(Models, save=outpath)

# Remove the smallest cluster (contains false positive spikes)
# TODO change for smallest amplitude?
if rm_smaller_cluster:
    remove_spikes(SpikeInfo,unit_column,'min')
    n_changes,Rss_sum,ScoresSum,units,AICs,n_units = eval_model(SpikeInfo,this_unit_col,prev_unit_col,Scores,Templates,ScoresSum,AICs)

    # plot templates and models for last column
    outpath = plots_folder / ("Templates_final%s" % ( fig_format))
    plot_templates(Templates, SpikeInfo, unit_column, save=outpath)
    outpath = plots_folder / ("Models_final%s" % (fig_format))
    plot_Models(Models, save=outpath)



"""
 
 ######## #### ##    ## ####  ######  ##     ## 
 ##        ##  ###   ##  ##  ##    ## ##     ## 
 ##        ##  ####  ##  ##  ##       ##     ## 
 ######    ##  ## ## ##  ##   ######  ######### 
 ##        ##  ##  ####  ##        ## ##     ## 
 ##        ##  ##   ###  ##  ##    ## ##     ## 
 ##       #### ##    ## ####  ######  ##     ## 
 
"""

last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]

# plot
outpath = plots_folder / ("Convergence_Rss" + fig_format)
plot_convergence(ScoresSum, save=outpath)

outpath = plots_folder / ("Convergence_AIC" + fig_format)
plot_convergence(AICs, save=outpath)

outpath = plots_folder / ("Clustering" + fig_format)
plot_clustering(Templates, SpikeInfo, last_unit_col, save=outpath)

# update spike labels
kernel = ele.kernels.GaussianKernel(sigma=kernel_fast * pq.s)
# it = its-1 # the last
it = it-1

for i, seg in tqdm(enumerate(Blk.segments),desc="populating block for output"):
    spike_labels = SpikeInfo.groupby(('segment')).get_group((i))['unit_%i' % it].values
    seg.spiketrains[0].annotations['unit_labels'] = list(spike_labels)

    # make spiketrains
    St = seg.spiketrains[0]
    spike_labels = St.annotations['unit_labels']
    sts = [St]

    for unit in units:
        times = St.times[sp.array(spike_labels) == unit]
        st = neo.core.SpikeTrain(times, t_start = St.t_start, t_stop=St.t_stop)
        st.annotate(unit=unit)
        sts.append(st)
    seg.spiketrains=sts

    # est firing rates
    asigs = [seg.analogsignals[0]]
    for unit in units:
        St, = select_by_dict(seg.spiketrains, unit=unit)
        frate = ele.statistics.instantaneous_rate(St, kernel=kernel, sampling_period=1/fs)
        frate.annotate(kind='frate_fast', unit=unit)
        asigs.append(frate)
    seg.analogsignals = asigs

# store SpikeInfo
outpath = results_folder / 'SpikeInfo.csv'
print_msg("saving SpikeInfo to %s" % outpath)
SpikeInfo.to_csv(outpath)

# store Block
outpath = results_folder / 'result.dill'
print_msg("saving Blk as .dill to %s" % outpath)
sssio.blk2dill(Blk, outpath)

print_msg("data is stored")

# output csv data
if Config.getboolean('output','csv'):
    print_msg("writing csv")

    # SpikeTimes
    for i, Seg in enumerate(Blk.segments):
        seg_name = Path(Seg.annotations['filename']).stem
        for j, unit in enumerate(units):
            St, = select_by_dict(Seg.spiketrains, unit=unit)
            outpath = results_folder / ("Segment_%s_unit_%s_spike_times.txt" % (seg_name, unit))
            np.savetxt(outpath, St.times.magnitude)

    # firing rates - full res
    for i, Seg in enumerate(Blk.segments):
        FratesDf = pd.DataFrame()
        seg_name = Path(Seg.annotations['filename']).stem
        for j, unit in enumerate(units):
            asig, = select_by_dict(Seg.analogsignals, kind='frate_fast', unit=unit)
            FratesDf['t'] = asig.times.magnitude
            FratesDf[unit] = asig.magnitude.flatten()

        outpath = results_folder / ("Segment_%s_frates.csv" % seg_name)
        FratesDf.to_csv(outpath)
            
        # firing rates - downsampled
        # tbins = sp.arange(0,12.2,0.1)
        # FratesDf_ds = pd.DataFrame(columns=FratesDf.columns)
        # for i in range(1,tbins.shape[0]):
        #     t0 = tbins[i-1]
        #     t1 = tbins[i]
        #     ix = sp.logical_and(FratesDf['t'] > t0,FratesDf['t'] < t1)
        #     FratesDf_ds = FratesDf_ds.append(FratesDf.iloc[ix.values].mean(axis=0),ignore_index=True)
        # FratesDf_ds['t'] = tbins[:-1]

        # outpath = results_folder / ("Segment_%s_frates_downsampled.csv" % seg_name)
        # FratesDf_ds.to_csv(outpath)
    

"""
 
 ########  ##        #######  ########    #### ##    ##  ######  ########  ########  ######  ######## 
 ##     ## ##       ##     ##    ##        ##  ###   ## ##    ## ##     ## ##       ##    ##    ##    
 ##     ## ##       ##     ##    ##        ##  ####  ## ##       ##     ## ##       ##          ##    
 ########  ##       ##     ##    ##        ##  ## ## ##  ######  ########  ######   ##          ##    
 ##        ##       ##     ##    ##        ##  ##  ####       ## ##        ##       ##          ##    
 ##        ##       ##     ##    ##        ##  ##   ### ##    ## ##        ##       ##    ##    ##    
 ##        ########  #######     ##       #### ##    ##  ######  ##        ########  ######     ##    
 
"""

# plot all sorted spikes
for j, Seg in enumerate(Blk.segments):
    seg_name = Path(Seg.annotations['filename']).stem
    outpath = plots_folder / (seg_name + '_overview' + fig_format)
    plot_segment(Seg, units, save=outpath)

# plot all sorted spikes
# zoom = sp.array(Config.get('output','zoom').split(','),dtype='float32') / 1000

max_window = 4 #AG: TODO add to config file
unit_column = last_unit_col

plot_fitted_spikes_complete(Blk, Models, SpikeInfo, unit_column, max_window, plots_folder, fig_format,wsize=n_samples)

max_window = 0.3 #AG: TODO add to config file
plot_fitted_spikes_complete(Blk, Models, SpikeInfo, unit_column, max_window, plots_folder, fig_format,wsize=n_samples)

max_window = 0.3 #AG: TODO add to config file
plot_fitted_spikes_complete(Blk, Templates, SpikeInfo, unit_column, max_window, plots_folder, fig_format,wsize=n_samples,extension='_templates')

print_msg("plotting done")
print_msg("all done - quitting")

sys.exit()
