import matplotlib.pyplot as plt
import pandas as pd
from sssio import * 
from plotters import *
from functions import *
from sys import path
from postprocessing_functions import *

import configparser
#import tracemalloc
import gc


plt.rcParams.update({'font.size': 6})

################################################################
##  
##  Some tools 
##
################################################################

def insert_row(df, idx, df_insert):
    dfA = df.iloc[:idx, ]
    dfB = df.iloc[idx:, ]

    df = dfA.append(df_insert).append(dfB).reset_index(drop = True)

    return df

def delete_row(df, idx):
    return df.iloc[:idx, ].append(df.iloc[idx+1:]).reset_index(drop = True)
    seg.analogsignals[0].sampling_rate
    
def insert_spike(SpikeInfo, new_column, i, o_spike, o_spike_time, o_spike_unit):
    idx= max(i,o_spike)
    SpikeInfo= insert_row(SpikeInfo, idx, SpikeInfo.iloc[i, ]) # insert a copy of row i 
    SpikeInfo[new_column][idx]= o_spike_unit
    SpikeInfo['id'][idx]= str(SpikeInfo['id'][idx])+'b'
    SpikeInfo['time'][idx]= o_spike_time
    SpikeInfo['good'][idx]= False   # do not use for building templates!
    SpikeInfo['frate_fast'][idx]= SpikeInfo['frate_'+o_spike_unit][idx]   # update rate_fast to the correct rate for the nwe spike's identity
    return SpikeInfo

    
################################################################
##  
##              Load clustering result
##
################################################################

# get config
config_path = Path(os.path.abspath(sys.argv[1]))
sssort_path = os.path.dirname(os.path.abspath(sys.argv[0]))
Config = configparser.ConfigParser()
Config.read(config_path)
print_msg('config file read from %s' % config_path)

# handling paths and creating outunits.sort()put directory
data_path = Path(Config.get('path', 'data_path'))
if not data_path.is_absolute():
    data_path = config_path.parent / data_path

exp_name = Config.get('path', 'experiment_name')
results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots' / 'post_process_all' 

os.makedirs(plots_folder, exist_ok=True)

Blk=get_data(results_folder/"result.dill")
SpikeInfo = pd.read_csv(results_folder/"SpikeInfo.csv")
nSpikeInfo= SpikeInfo.copy()

unit_column = 'unit_labeled'
SpikeInfo = SpikeInfo.astype({unit_column: str})
units = get_units(SpikeInfo, unit_column)

#plot config
plotting_changes = Config.getboolean('postprocessing','plot_changes')
mpl.rcParams['figure.dpi'] = Config.get('output','fig_dpi')
fig_format = Config.get('output','fig_format')

stimes= SpikeInfo['time']
Seg = Blk.segments[0]
fs= Seg.analogsignals[0].sampling_rate
ifs= int(fs/1000)   # sampling rate in kHz as integer value to convert ms to bins

# recalculate the latest firing rates according to spike assignments in unit_column
#kernel_slow = Config.getfloat('kernels','sigma_slow')
kernel_fast = Config.getfloat('kernels','sigma_fast')
calc_update_final_frates(Blk.segments, nSpikeInfo, unit_column, kernel_fast)

templates_path = config_path.parent / results_folder / "Templates_ini.npy"
Templates= np.load(templates_path)
print_msg('templates read from %s' % templates_path)
n_model_comp = Config.getint('spike model','n_model_comp')
Models = train_Models(nSpikeInfo, unit_column, Templates, n_comp=n_model_comp, verbose=True, model_type= Spike_Model_Nlin)

unit_ids= nSpikeInfo[unit_column]
units = get_units(nSpikeInfo, unit_column)
frate= {}
for unit in units:
    frate[unit]= nSpikeInfo['frate_'+unit]

sz_wd= Config.getfloat('postprocessing','spike_window_width')
align_mode= Config.get('postprocessing','vertical_align_mode')
same_spike_tolerance= Config.getfloat('postprocessing','spike_position_tolerance')
same_spike_tolerance= int(same_spike_tolerance*ifs) # in time steps
d_accept= Config.getfloat('postprocessing','max_dist_for_auto_accept')
d_reject= Config.getfloat('postprocessing','min_dist_for_auto_reject')
min_diff= Config.getfloat('postprocessing','min_diff_for_auto_accept')
max_spike_diff= int(Config.getfloat('postprocessing','max_compound_spike_diff')*ifs)
n_samples= np.array(Config.get('spike model','template_window').split(','),dtype='float32')/1000.0
n_samples= np.array(n_samples*fs, dtype= int)
try:
    spkr = sp.array(Config.get('postprocessing','spike_range').split(','),dtype='int')
    spike_range= range(spkr[0], spkr[1])
except:
    spike_range = range(1,len(unit_ids)-1)
    
asig= Seg.analogsignals[0]
asig= asig.reshape(asig.shape[0])
asig= align_to(asig,align_mode)
as_min= np.amin(asig)
as_max= np.amax(asig)
y_lim= [ 1.05*as_min, 1.05*as_max ]
n_wd= int(sz_wd*ifs)
n_wdh= n_wd//2

new_column = 'unit_final'
if new_column not in nSpikeInfo.keys():
    nSpikeInfo[new_column]= SpikeInfo[unit_column][:]
offset= 0   # will keep track of shifts due to inserted and deleted spikes 
# don't consider first and last spike to avoid corner cases; these do not matter in practice anyway
#tracemalloc.start()
skip= False
for i in spike_range:
    if skip:
        skip= False
        continue
    start= int((float(stimes[i])*1000-sz_wd/2)*ifs)
    stop= start+n_wd
    if (start > 0) and (stop < len(asig)):   # only do something if the spike is not too close to the start or end of the recording, otherwise ignore
        v= np.array(asig[start:stop],dtype= float)
        v= align_to(v,align_mode)
        d= []
        sh= []
        un= []
        templates= {}
        for unit in units[:2]:
            templates[unit]= make_single_template(Models[unit], frate[unit][i])
            templates[unit]= align_to(templates[unit],align_mode)
            for pos in range(n_wdh-same_spike_tolerance,n_wdh+same_spike_tolerance):
                d.append(dist(v,templates[unit],n_samples,pos))
                sh.append(pos)
                un.append(unit)
        d2= []
        sh2= []
        for pos1 in range(n_wd):
            for pos2 in range(n_wd):
                # one of the spikes must be close to the spike time under consideration
                if ((abs(pos1-n_wdh) <= same_spike_tolerance) or (abs(pos2-n_wdh) <= same_spike_tolerance)) and (abs(pos1-pos2) < max_spike_diff):
                    d2.append(compound_dist(v,templates['a'],templates['b'],n_samples,pos1,pos2))
                    sh2.append((pos1,pos2))

        # work out the final decision
        best= np.argmin(d)
        best2= np.argmin(d2)
        d_min= min(d[best],d2[best2])
        choice= 1 if d[best] <= d2[best2] else 2
        d_diff= abs(d[best]-d2[best2])
        print_msg("Single spike d={}, compound spike d={}, difference={}".format(d[best], d2[best2], d_diff))
        # show some plots first
        zoom= (float(stimes[i])-sz_wd/1000*20,float(stimes[i])+sz_wd/1000*20)
        if d_min >= d_accept or 200*d_diff/(d[best]+d2[best2]) < min_diff:
            # make plots and save them
            fig2, ax2= plot_postproc_context(Seg, 0, Models, nSpikeInfo, new_column, zoom=zoom, box= (float(stimes[i]),sz_wd/1000), wsize= n_samples, ylim= y_lim)
            outpath = plots_folder / (str(i)+'_context_plot' + fig_format)
            fig2.savefig(outpath)
            fig, ax= plt.subplots(ncols=2, sharey= True, figsize=[ 4, 2])
            dist(v,templates[un[best]],n_samples,sh[best],unit= un[best],ax= ax[0])
            ax[0].set_ylim(y_lim)
            compound_dist(v,templates['a'],templates['b'],n_samples,sh2[best2][0],sh2[best2][1],ax[1])
            ax[1].set_ylim(y_lim)
            outpath = plots_folder / (str(i)+'_template_matches' + fig_format)
            fig.savefig(outpath)
            if d_min > d_reject:
                choice= 0
            elif (d_min >= d_accept or (200*d_diff/(d[best]+d2[best2]) < min_diff)):
                # ask user
                fig2.show()
                fig.show()
                if (200*d_diff/(d[best]+d2[best2]) <= min_diff):
                    reason= "two very close matches"
                elif d_min >= d_accept:
                    reason= "no good match but not bad enough to reject"
                print("User feedback required: "+reason)
                choice= int(input("Single spike (1), Compound spike (2), no spike (0)? "))
            plt.close(fig2)
            plt.close(fig)
        # apply choice 
        if choice == 1:
            # it;s a single spike - choose the appropriate single spike unit
            spike_time= stimes[i]-n_wdh/1000/ifs+sh[best]/1000/ifs  # spike time in seconds
            if (abs(stimes[i-1]-spike_time)*1000*ifs < same_spike_tolerance) and nSpikeInfo[new_column][i-1+offset] == un[best]:
                # this spikes is already recorded with the same type
                print_msg("Spike {}: time= {}: Single spike, was type {} but already exists as spike {}; marked for deletion (-3)".format(i,stimes[i],SpikeInfo[unit_column][i],nSpikeInfo['id'][i-1+offset]))
                nSpikeInfo[new_column][i+offset]= -3
                nSpikeInfo['good'][i+offset]= False
                nSpikeInfo['frate_fast'][i+offset]= nSpikeInfo['frate_'+un[best]][i+offset]
            else:
                print_msg("Spike {}: time= {}: Single spike, was type {}, now  of type {}, time= {}".format(i,stimes[i],SpikeInfo[unit_column][i],un[best],spike_time))
                nSpikeInfo[new_column][i+offset]= un[best]
                nSpikeInfo['time'][i+offset]= spike_time
                nSpikeInfo['frate_fast'][i+offset]= nSpikeInfo['frate_'+un[best]][i+offset]
        elif choice == 2:
            # it's a compound spike - choose the appropriate spike unit and handle second spike
            orig_spike= np.argmin(abs(np.array(sh2[best2])-n_wdh))
            other_spike= 1-orig_spike
            spike_unit= 'a' if orig_spike == 0 else 'b'
            spike_time= stimes[i]-n_wdh/1000/ifs+sh2[best2][orig_spike]/1000/ifs  # spike time in seconds
            print_msg("Spike {}: time= {}: Compound spike, first spike of type {}, time= {}".format(i,SpikeInfo['time'][i],spike_unit,spike_time))
            nSpikeInfo[new_column][i+offset]= spike_unit
            nSpikeInfo['time'][i+offset]= spike_time
            nSpikeInfo['good'][i+offset]= False   # do not use compound spikes for Model building
            nSpikeInfo['frate_fast'][i+offset]= nSpikeInfo['frate_'+spike_unit][i+offset]
            if sh2[best2][other_spike] < sh2[best2][orig_spike]:
                o_spike_id= i-1
            else:
                o_spike_id= i+1
                skip= True
            o_spike_unit= 'a' if other_spike == 0 else 'b'
            o_spike_time= stimes[i]-n_wdh/1000/ifs+sh2[best2][other_spike]/1000/ifs  # spike time in seconds
            if abs(stimes[o_spike_id]-o_spike_time)*1000*ifs < same_spike_tolerance:
                # the other spike coincides with the previous spike in the original list
                # make sure that the previous decision is consistent with the current one
                print_msg("Spike {}: time= {}: Compound spike, second spike was known as {}, now of type {}, time= {}".format(i,SpikeInfo['time'][i],SpikeInfo[unit_column][o_spike_id],o_spike_unit,o_spike_time))
                nSpikeInfo[new_column][o_spike_id+offset]= o_spike_unit
                nSpikeInfo['good'][o_spike_id+offset]= False   # do not use compound spikes for Model building
                nSpikeInfo['frate_fast'][o_spike_id+offset]= nSpikeInfo['frate_'+o_spike_unit][o_spike_id+offset]
                    
            else:
                # the other spike does not yet exist in the list: insert new row
                print_msg("Spike {}: Compound spike, second spike was undetected, inserted new spike of type {}, time= {}".format(i,SpikeInfo['time'][i],o_spike_unit,o_spike_time))
                nSpikeInfo= insert_spike(nSpikeInfo, new_column, i+offset, o_spike_id+offset, o_spike_time, o_spike_unit)
                offset+= 1

               
        else:
            # it's a non-spike - delete it
            #nSpikeInfo= delete_row(nSpikeInfo, i+offset)
            nSpikeInfo[new_column][i+offset]= '-3'
            nSpikeInfo['good'][i+offset]= False   # definitively do not use for model building
            print_msg("Spike {}: Not a spike, marked for deletion (-3)".format(i))
            #offset-= 1
        if skip:
            i= i+1
            
calc_update_final_frates(Blk.segments, nSpikeInfo, unit_column, kernel_fast)

# Saving
kernel = ele.kernels.GaussianKernel(sigma=kernel_fast * pq.s)
for i, seg in tqdm(enumerate(Blk.segments),desc="populating block for output"):
    fs = seg.analogsignals[0].sampling_rate
    spike_labels = nSpikeInfo.groupby(('segment')).get_group((i))[new_column].values
    times= nSpikeInfo.groupby(('segment')).get_group((i))['time'].values
    St = seg.spiketrains[0]
    seg.spiketrains[0]= neo.core.SpikeTrain(times, units='sec', t_start = St.t_start,t_stop=St.t_stop)
    seg.spiketrains[0].array_annotate(unit_labels= list(spike_labels))
    
    # make spiketrains
    St = seg.spiketrains[0]
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


#save all
units = get_units(SpikeInfo,unit_column)
print_msg("Number of spikes in trace: %d"%SpikeInfo[unit_column].size)
print_msg("Number of bad spikes: %d"%len(SpikeInfo.groupby(['good']).get_group(True)[unit_column]))
# print_msg("Number of good spikes: %d"%len(SpikeInfo.groupby(['good']).get_group(False)[unit_column]))
print_msg("Number of clusters: %d"%len(units))

output_csv = Config.getboolean('output', 'csv')
# warning firing rates not saved, too high memory use.
save_all(results_folder, output_csv, nSpikeInfo, Blk, units, Frates=False)

seg_name = Path(Seg.annotations['filename']).stem
outpath = plots_folder / (seg_name + '_overview' + fig_format)
plot_segment(Seg, units, save=outpath)


max_window= Config.getfloat('output','max_window_fitted_spikes_overview')
plot_fitted_spikes_complete(Blk, Models, nSpikeInfo, new_column, max_window, plots_folder, fig_format,wsize=n_samples,extension='_templates')


"""
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)
"""
