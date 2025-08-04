#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import glob
import h5py
import read_ud3
import scipy.signal.windows as sw

def rtdi(fname,t_mess=10,n_int=12):
    """
    t_mess, how many seconds to read
    n_int, how many spectra to integrate incoherently
    """
    r=read_ud3.UD3_Reader(fname,t_mess=t_mess)
    d=r.get_next_chunk()
    t0=r.time
    n_chan=d.shape[0]
    n_range=d.shape[1]
    n_time=d.shape[2]
    wf=sw.hann(n_time)
    eff_prf=r.PRF/r.n_integrations
    dtau=(1.0/eff_prf)*d.shape[2]
    dops=n.fft.fftshift(n.fft.fftfreq(n_time,d=1/eff_prf))
    ranges=(r.deltaR)*n.arange(n_range)+r.range_offset
    for k in range(int(r.n_chunks/n_int)):
        S=n.zeros([n_range,n_time])
        for ii in range(n_int):
            for chi in range(n_chan):
                for rg in range(n_range):
                    S[rg,:]+=n.fft.fftshift(n.abs(n.fft.fft(wf*d[chi,rg,:]))**2.0)
            d=r.get_next_chunk()
#            print(t0)
        tdata=dtau*k*n_int + t0
        #print(tdata)
        dB=10*n.log10(S)
        nf=n.nanmedian(dB)
        plt.pcolormesh(dops,ranges,dB-nf,vmin=0,vmax=10)#n.abs(d[0,:,:]),vmin=0,vmax=2000)
        plt.xlabel("Doppler shift (Hz)")
        plt.ylabel("Range (km)")
        cb=plt.colorbar()
        cb.set_label("Power (dB)")
        plt.title("Range-Doppler Intensity\n%s"%(n.datetime64(int(tdata),"s")))
        plt.tight_layout()
        plt.savefig("rtdi-%06d.png"%(int(tdata)))
        print("writing to file: rtdi-%06d.png"%(int(tdata)))

        plt.clf()
        plt.close()
#        plt.show()
#        print(n_chan)
 #       print(n_range)
  #      print(n_time)



# edit this to point to the filename that you want to process
rtdi("raw.bdumr.20250723_090938.ud3")