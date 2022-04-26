import matplotlib.pyplot as plt
import dill
from sssio import *
import os
import sys

in_path = os.path.abspath(sys.argv[1])
out_path = os.path.abspath(sys.argv[2])
col = int(sys.argv[3])

# # read data
# with neo.NixIO(filename=in_path) as Reader:
#     print("reading neo.block from neo *nix file %s" % in_path)
#     Blk = Reader.read_block()
# with np.readtxt(filename=in_path) as Reader:
#     print("reading neo.block from neo *nix file %s" % in_path)
#     Blk = Reader.read_block()
# data = asc2seg(in_path)
# seg_t
# segment = raw2seg(in_path, 10000, float, scale=0.001)
segment = asc2seg_noheader(in_path, 10000, unit=pq.mV, header_rows=2, col=col)

Blk = neo.core.Block()
Blk.segments = [segment]


print(len(Blk.segments[0].analogsignals))

plt.plot(Blk.segments[0].analogsignals[0])
plt.show()

with open(out_path, 'wb') as fH:
    print("dumping neo.block to dill file %s" % out_path)
    dill.dump(Blk, fH)
