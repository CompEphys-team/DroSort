import matplotlib.pyplot as plt
import pandas as pd
from sssio import * 
from plotters import *
from functions import *
from sys import path
from postprocessing_functions import *

import configparser

################################################################
##  
##  Some tools 
##
################################################################

def insert_row(df, idx, df_insert):
    print(idx)
    print(df_insert)
    dfA = df.iloc[:idx, ]
    dfB = df.iloc[idx:, ]

    df = dfA.append(df_insert).append(dfB).reset_index(drop = True)

    return df

def delete_row(df, idx):
    return df.iloc[:idx, ].append(df.iloc[idx+1:]).reset_index(drop = True)
    
def insert_spike(SpikeInfo, new_column, i, o_spike, o_spike_time, o_spike_unit):
    idx= max(i,o_spike)
    SpikeInfo= insert_row(SpikeInfo, idx, SpikeInfo.iloc[i, ]) # insert a copy of row i 
    SpikeInfo[new_column][idx]= o_spike_unit
    SpikeInfo['id'][idx]= str(SpikeInfo['id'][idx])+'b'
    SpikeInfo['time'][idx]= o_spike_time
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

unit_column = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
SpikeInfo = SpikeInfo.astype({unit_column: str})
units = get_units(SpikeInfo, unit_column)

#plot config
plotting_changes = Config.getboolean('postprocessing','plot_changes')
mpl.rcParams['figure.dpi'] = Config.get('output','fig_dpi')
fig_format = Config.get('output','fig_format')

stimes= SpikeInfo['time']
Seg = Blk.segments[0]

# recalculate the latest firing rates according to spike assignments in unit_column
kernel_slow = Config.getfloat('kernels','sigma_slow')
kernel_fast = Config.getfloat('kernels','sigma_fast')
calc_update_frates(Blk.segments, SpikeInfo, unit_column, kernel_fast, kernel_slow)

templates_path = config_path.parent / results_folder / "Templates_ini.npy"
Templates= np.load(templates_path)
print_msg('templates read from %s' % templates_path)
n_model_comp = Config.getint('spike model','n_model_comp')
Models = train_Models(SpikeInfo, unit_column, Templates, n_comp=n_model_comp, verbose=True)

unit_ids= SpikeInfo[unit_column]
units = get_units(SpikeInfo, unit_column)
frate= {}
for unit in units:
    frate[unit]= SpikeInfo['frate_from_'+unit]

sz_wd= Config.getfloat('postprocessing','spike_window_width')
align_mode= Config.get('postprocessing','vertical_align_mode')
same_spike_tolerance= Config.getfloat('postprocessing','spike_position_tolerance')
same_spike_tolerance= int(same_spike_tolerance*10) # in time steps
d_accept= Config.getfloat('postprocessing','max_dist_for_auto_accept')
min_diff= Config.getfloat('postprocessing','min_diff_for_auto_accept')

asig= Seg.analogsignals[0]
asig= asig.reshape(asig.shape[0])
asig= align_to(asig,align_mode)
n_wd= int(sz_wd*10)
n_wdh= n_wd//2

new_column = 'unit_final'
empty= [""] * len(SpikeInfo.index)
nSpikeInfo[new_column]= empty
offset= 0   # will keep track of shifts due to inserted and deleted spikes 
# don't consider first and last spike to avoid corner cases; these do not matter in practice anyway
for i in range(1,len(unit_ids)-1):
    start= int((float(stimes[i])*1000-sz_wd/2)*10)
    stop= start+n_wd
    if (start > 0) and (stop < len(asig)):   # only do something if the spike is not too close to the start or end of the recording, otherwise ignore
        v= np.array(asig[start:stop],dtype= float)
        d= []
        sh= []
        un= []
        templates= {}
        for unit in units[:2]:
            templates[unit]= make_single_template(Models[unit], frate[unit][i])
            templates[unit]= align_to(templates[unit],align_mode)
            for pos in range(n_wdh-same_spike_tolerance,n_wdh+same_spike_tolerance):
                d.append(dist(v,templates[unit],pos))
                sh.append(pos)
                un.append(unit)
        #print(d)
        #print(sh)
        #print(un)
        d2= []
        sh2= []
        for pos1 in range(n_wd):
            for pos2 in range(n_wd):
                # one of the spikes must be close to the spike time under consideration
                if (abs(pos1-n_wdh) <= same_spike_tolerance) or (abs(pos2-n_wdh) <= same_spike_tolerance):
                    d2.append(compound_dist(v,templates['a'],templates['b'],pos1,pos2))
                    sh2.append((pos1,pos2))

        # work out the final decision
        best= np.argmin(d)
        best2= np.argmin(d2)
        d_min= min(d[best],d2[best2])
        choice= 0 if d[best] <= d2[best2] else 1
        d_diff= abs(d[best]-d2[best2])
        print(d_min, d_diff)
        if d_min < d_accept and d_diff > min_diff:
            pass
        else:
            # ask user
            # show some plots first
            zoom= (float(stimes[i])-sz_wd/1000*20,float(stimes[i])+sz_wd/1000*20)
            fig2, ax2= plot_postproc_context(Seg, 0, Models, SpikeInfo, unit_column, zoom=zoom, box= (float(stimes[i]),sz_wd/1000))
            fig2.show()
            fig, ax= plt.subplots(ncols=2, sharey= True)
            fig.show()
            s_best_d= dist(v,templates[un[best]],sh[best],ax[0])
            c_best_d= compound_dist(v,templates['a'],templates['b'],sh2[best2][0],sh2[best2][1],ax[1])
        
            outpath = plots_folder / (str(i)+'_template_matches' + fig_format)
            fig.savefig(outpath)
            reason= "no good match" if d_min >= d_accept else "two very close matches"
            print("User feedback required: "+reason)
            choice= int(input("Single spike (0), Compound spike (1), no spike (2)?"))
            plt.close(fig2)
            plt.close(fig)
        # apply choice 
        if choice == 0:
            # it;s a single spike - choose the appropriate single spike unit
            nSpikeInfo[new_column][i+offset]= un[best]
        elif choice == 1:
            # it's a compound spike - choose the appropriate spike unit and handle second spike
            orig_spike= np.argmin(abs(np.array(sh2[best2])-n_wdh))
            other_spike= 1-orig_spike
            nSpikeInfo[new_column][i+offset]= 'a' if orig_spike == 0 else 'b'
            if sh2[best2][other_spike] < n_wdh:
                o_spike_id= i-1
            else:
                o_spike_id= i+1
            o_spike_unit= 'a' if other_spike == 0 else 'b'
            o_spike_time= stimes[i]-n_wdh/10000+sh2[best2][other_spike]/10000
            # the other spike is earlier
            if abs(stimes[o_spike_id]-o_spike_time)*10000 < same_spike_tolerance:
                # the other spike coincides with the previous spike in the original list
                # make sure that the previous decision is consistent with the current one
                print(SpikeInfo[unit_column][o_spike_id])
                print(o_spike_unit)
                assert((SpikeInfo[unit_column][o_spike_id] == o_spike_unit) or (SpikeInfo[unit_column][o_spike_id] == '-2'))
            else:
                # the other spike does not yet exist in the list: insert new row
                nSpikeInfo= insert_spike(nSpikeInfo, new_column, i, o_spike_id, o_spike_time, o_spike_unit)
                offset+= 1
        else:
            # it's a non-spike - delete it
            nSpikeInfo= delete_row(nSpikeInfo, i)
            offset-= 1
        nSpikeInfo.to_csv(results_folder/"nSpikeInfo.csv")
nSpikeInfo.to_csv(results_folder/"SpikeInfo.csv")
        

