#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import read_ud3
import h5py
import scipy.constants as c
import stuffr

def remove_ionosonde(d):
    for i in range(d.shape[0]):
        med_pwr=n.median(n.abs(d[i,:,:]),axis=0)
        avg_noise=n.median(med_pwr)
        std_est=n.median(n.abs(med_pwr-avg_noise))
        bad_idx=n.where(med_pwr-avg_noise > 5.0*std_est)[0]
        d[i,:,bad_idx]=0.0
    return(d)


def xc_incoh(fname="/data3/SLICE_RAW/MRRAW_20171220/raw.sodankyla.5a39ae21",
             t_mess=1.0,
             alias_num=2.0,
             max_n_chunks=None):

    print("read")
    r=read_ud3.UD3_Reader(fname,t_mess=t_mess)
    print(r.n_chan)
    print(r.antenna_coords)
    d=r.get_next_chunk()
    print(r.n_chunks)
    n_chan=d.shape[0]
    n_range=d.shape[1]
    n_time=d.shape[2]
    print(n_time)
    freqs=n.fft.fftshift(n.fft.fftfreq(d.shape[2],d=r.IPP*r.n_integrations))
    vels=freqs*c.c/2.0/(r.frequency*1e6)

    ipp_range=alias_num*r.IPP*c.c/2.0/1e3
    
    ranges=ipp_range + r.deltaR*n.arange(d.shape[1])+r.range_offset
    
    t0=r.time
    eff_ipp=2144.0/r.n_integrations
    dtau=(1.0/eff_ipp)*d.shape[2]
    r=read_ud3.UD3_Reader(fname,t_mess=t_mess)
    print(d.shape)


    if max_n_chunks == None:
        n_c=r.n_chunks
    else:
        n_c=max_n_chunks
    RTI=n.zeros([n_c,n_range],dtype=n.float32)
    DOP=n.zeros([n_c,n_range],dtype=n.float32)
        
    tvec=[]
    for k in range(n_c):
        print("%d/%d"%(k,n_c))
        S=n.zeros([d.shape[1],d.shape[2]])
        d=r.get_next_chunk()
        print(d.shape)
        tvec.append(r.time + dtau*k)
        for i in range(n_chan):
            for j in range(n_range):
                z=d[i,j,:]
                z=z-n.mean(z)
                S[j,:]+=n.fft.fftshift(n.abs(n.fft.fft(d[i,j,:]))**2.0)
        for j in range(n_range):
            RTI[k,j]=n.max(S[j,:])
            DOP[k,j]=vels[n.argmax(S[j,:])]
    tvec=n.array(tvec)
    return(tvec,ranges,RTI,DOP)

if __name__ == "__main__":
    tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
                                      max_n_chunks=None)
    ho=h5py.File("out.h5","w")
    ho["tvec"]=tvec
    ho["ranges"]=ranges
    ho["RTI"]=RTI
    ho["DOP"]=DOP
    ho.close()
    
    plt.pcolormesh(tvec,ranges,n.transpose(10.0*n.log10(RTI)),cmap="plasma")
    plt.xlabel("Time (s)")
    plt.ylabel("Range (km)")
    plt.colorbar()
    plt.show()
    plt.pcolormesh(tvec,ranges,n.transpose(DOP),cmap="seismic",vmin=-100,vmax=100.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Range (km)")    
    plt.colorbar()
    plt.show()
