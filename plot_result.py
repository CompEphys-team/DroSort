import matplotlib.pyplot as plt
import pandas as pd
from sssio import * 
from plotters import *
from functions import *

# path = sys.argv[1]

Blk=get_data(sys.argv[1]+"/result.dill")


SpikeInfo = pd.read_csv(sys.argv[1]+"/SpikeInfo.csv")

print(SpikeInfo.describe())

last_unit_col = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]

units = get_units(SpikeInfo,last_unit_col)

colors = get_colors(units)

for seg in Blk.segments:
    print(seg)
    for n,asig in enumerate(seg.analogsignals):
        plt.subplot(len(seg.analogsignals),1,n+1)
        plt.plot(asig.times,asig.magnitude)

        if n==0:
            for i,sp in enumerate(seg.spiketrains[0]):
                unit =SpikeInfo[SpikeInfo['id']==i][last_unit_col].values[0] 
                col = colors[str(unit)]

                plt.plot(seg.spiketrains[0].times[i],seg.spiketrains[0].waveforms.reshape(seg.spiketrains[0].times.size)[i],'.',color=col)

plt.show()
