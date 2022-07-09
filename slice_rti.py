#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import glob
import h5py

fl=glob.glob("/data3/SLICE_RAW/20s/rd*.h5")
fl.sort()

n_files=len(fl)
print(n_files)
h=h5py.File(fl[0],"r")
print(h.keys())
S=n.zeros([n_files,h["S"].shape[0]])
zidx=n.where(h["vels"].value==0)[0]
ranges=n.copy(h["range"].value)
h.close()

tvec=n.zeros(n_files)
for i in range(n_files):
    h=h5py.File(fl[i],"r")
    SS=h["S"].value
    SS[:,zidx]=0.0 # dc gc removal
    S[i,:]=n.max(SS,axis=1)
    tvec[i]=h["t0"].value
    h.close()
dB=10.0*n.log10(S)
dB=dB-n.nanmedian(dB)

print(n.max(tvec)-n.min(tvec))
plt.pcolormesh(tvec,ranges,n.transpose(dB),vmin=-2,vmax=10,cmap="viridis")
plt.colorbar()
plt.ylabel("Range (km)")
plt.xlabel("Time (s since 1970)")
plt.show()


