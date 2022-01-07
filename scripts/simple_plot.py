# sys
import sys
import os
import copy
import dill
import shutil
import configparser
from pathlib import Path
from tqdm import tqdm


# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# own
from functions import *
from plotters import *
import sssio


data_path = Path(os.path.abspath(sys.argv[1]))
data_path = sys.argv[1]


Blk = sssio.get_data(data_path)
Blk.name = 'test'
print_msg('data read from %s' % data_path)



for seg in Blk.segments:
	for asig in seg.analogsignals:
		plt.plot(asig.times,asig.magnitude)
		plt.show()