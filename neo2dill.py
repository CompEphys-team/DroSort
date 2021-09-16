import neo
import dill
import os
import sys

in_path= os.path.abspath(sys.argv[1])
out_path= os.path.abspath(sys.argv[2])

# read data
with neo.NixIO(filename=in_path) as Reader:
    print("reading neo.block from neo *nix file %s" % in_path)
    Blk = Reader.read_block()

with open(out_path, 'wb') as fH:
    print("dumping neo.block to dill file %s" % out_path)
    dill.dump(Blk, fH)
