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

# ephys
import neo
import elephant as ele

# own
from tools.functions import *
from tools.plotters import *
import tools.sssio as sssio

# banner
print(banner)

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

# get segment to analyse
seg_no= Config.getint('general','segment_number')

# handling paths and creating output directory
data_path = Path(Config.get('path','data_path'))
if not data_path.is_absolute():
    data_path = config_path.parent / data_path

exp_name = Config.get('path','experiment_name')
results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots' / 'spike_detection'

if os.path.isdir(plots_folder):
    # remove old plots if desired
    if input("Remove any existing plots in {} (Y/N)? ".format(plots_folder)).upper() == 'Y':
        os.system("rm %s/*"%plots_folder)
   
os.makedirs(plots_folder, exist_ok=True)
os.chdir(config_path.parent / exp_name)

# copy config
shutil.copyfile(config_path, config_path.parent / exp_name / config_path.name)

# read data
Blk = sssio.get_data(data_path)
Blk.name = exp_name
seg= Blk.segments[seg_no]
print_msg('data read from %s, analysing segment %d' % (data_path, seg_no))

# plotting
mpl.rcParams['figure.dpi'] = Config.get('output','fig_dpi')
fig_format = Config.get('output','fig_format')

#TODO restriction if you do not want to analyze the whole trace
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

seg.analogsignals[0].annotate(kind='original')

# highpass filter
freq = Config.getfloat('preprocessing','highpass_freq')
print_msg("highpass filtering data at %.2f Hz" % freq)
seg.analogsignals[0] = ele.signal_processing.butter(seg.analogsignals[0], highpass_freq=freq)
if 'filename' not in seg.annotations:
    seg.annotations['filename']= 'segment_'+str(seg_no)

# invert if peaks are negative
if Config.get('preprocessing','peak_mode') == 'negative':
    print_msg("inverting signal")
    seg.analogsignals[0] *= -1

if Config.getboolean('preprocessing','z_trials'):
    print_msg("z-scoring analogsignals for each trial")
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

min_ampl = Config.getfloat('preprocessing', 'min_amplitude')
max_dur = Config.getfloat('preprocessing', 'max_duration')

r_non_spikes = Config.getboolean('preprocessing','reject_non_spikes')

AnalogSignal, = select_by_dict(seg.analogsignals, kind='original')
bounds = [MAD(AnalogSignal)*mad_thresh, sp.inf] * AnalogSignal.units  # TN: Isn't this applying AnalogSignal.units twice (in MAD and here explicitly)?
bounds_neg = [MAD(AnalogSignal)*(mad_thresh-2), sp.inf] * AnalogSignal.units #TODO hardcode mad_thresh lower

fs = seg.analogsignals[0].sampling_rate
n_samples = (wsize * fs).simplified.magnitude.astype('int32')

if Config.get('preprocessing','peak_mode') == 'double':
    st = double_spike_detect(AnalogSignal, bounds, bounds_neg,wsize=n_samples)
else:
    st = spike_detect(AnalogSignal, bounds,wsize=n_samples)

if st.times.shape[0] == 0:
    stim_name = Path(seg.annotations['filename']).stem
    print_msg("no spikes found for segment %i:%s" % (i,stim_name))

st.annotate(kind='all_spikes')

n_spikes = st.shape[0]
print_msg("number of spikes found: %s" % n_spikes)
  
# remove border spikes
wsize = Config.getfloat('spike detect', 'wsize') * pq.ms
st_cut = st.time_slice(st.t_start + wsize/2, st.t_stop - wsize/2)
st_cut.t_start = st.t_start

# remove bad detections
if r_non_spikes:
    verbose = Config.getboolean('spike detect','verbose')
    st_cut,rejs = reject_non_spikes(AnalogSignal,st_cut,n_samples,min_ampl=min_ampl,max_dur=max_dur,verbose=verbose,plot=False)

seg.spiketrains.append(st_cut)

n_spikes = sp.sum([seg.spiketrains[0].shape[0] for seg in Blk.segments])
print_msg("total number of spikes found: %s" % n_spikes)

#Plot detected spikes
namepath = plots_folder / ("first_spike_detection_%d"%seg_no)
plot_spike_events(seg,thres=MAD(AnalogSignal)*mad_thresh,save=namepath,save_format=fig_format,show=False,max_window=0.4,max_row=3,rejs=rejs)

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

fs = seg.analogsignals[0].sampling_rate
n_samples= np.array(Config.get('spike model','template_window').split(','),dtype='float32')/1000.0
n_samples= np.array(n_samples*fs, dtype= int)
templates = []
data = seg.analogsignals[0].magnitude.flatten()
inds = (seg.spiketrains[0].times * fs).simplified.magnitude.astype('int32')
templates.append(get_Templates(data, inds, n_samples))

Templates = sp.concatenate(templates,axis=1)

#Save events and plot
plt.close()
zoom = sp.array(Config.get('output','zoom').split(','),dtype='float32') / 1000
outpath = plots_folder / ("templates_in_signal_init_%d"%seg_no+fig_format)
plot_templates_on_trace(seg, seg_no, Templates, save=outpath,wsize=n_samples,zoom=zoom)

#Save all to disk

# templates to disk
outpath = results_folder / 'Templates_ini.npy'
sp.save(outpath, Templates)
print_msg("saving Templates to %s" % outpath)

# rejected spikes to disk
outpath = results_folder / 'rejected_spikes.npy'
sp.save(outpath, rejs)
print_msg("saving rejected_spikes to %s" % outpath)

# store Block
outpath = results_folder / 'spikes_result.dill'
print_msg("saving Blk as .dill to %s" % outpath)
sssio.blk2dill(Blk, outpath)
