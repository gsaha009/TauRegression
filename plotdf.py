import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='PlottingDF')
parser.add_argument('-in',
                    '--input',
                    type=str,
                    required=True,
                    help="input h5 file")
parser.add_argument('-out',
                    '--output',
                    type=str,
                    required=True,
                    help="output file name")

args = parser.parse_args()


_df = args.input
df = pd.read_hdf(_df)

keys = list(df.keys())
nkeys = len(keys)
ncols = 10
nrows = int(np.ceil(nkeys/10))


plt.figure(figsize=(4*ncols,2*nrows))
for idx, key in enumerate(keys):
    idx=idx+1
    ax = plt.subplot(nrows,ncols,idx)
    arr = df[key]

    ax.hist(arr, 50, histtype="stepfilled", alpha=0.7, log=False)

    ax.set_title(f"{key}")
    ax.set_xlabel(f"{key}")
    #ax.legend()
plt.tight_layout()
plt.savefig(f'{args.outout}.png', dpi=300)
