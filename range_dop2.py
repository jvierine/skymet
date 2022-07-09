#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import read_ud3
import h5py
import scipy.constants as c
import stuffr
import scipy.signal as ss
import re
import os

def remove_ionosonde(d,
                     t,
                     tbad=[0.8,1.525,2.35,3.4,4.68,6.45,8.9,12.6],
                     ts=0.1):
    w = n.zeros(len(t),dtype=n.float32)
    for t0 in tbad:
        w+=n.exp(-(t-t0)**2.0/(2.0*ts**2.0))
    for ti in range(d.shape[2]):
        d[:,:,ti]=d[:,:,ti]*(1.0-w[ti])
    return(d)

def snr_r(S):
    SNR=n.copy(S)
    for ri in range(SNR.shape[0]):
        nfloor=n.nanmedian(SNR[ri,:])
        SNR[ri,:]=(SNR[ri,:]-nfloor)/(nfloor+1e-6)
    return(SNR)

def snr_0(S):
    SNR=n.copy(S)
    nfloor=n.nanmedian(SNR)
    SNR=(SNR-nfloor)/(nfloor+1e-6)
    return(SNR)

def clean_rti(S,clen=7,snr_min=0.5):
    CS=n.copy(S)
    CS[:,:]=0.0
#    nfloor=n.nanmedian(S)
 #   print(nfloor)
  #  SNR=(S-nfloor)/nfloor
    
    SNR=n.copy(S)
    
    for ri in range(SNR.shape[0]):
        nfloor=n.nanmedian(SNR[ri,:])
        SNR[ri,:]=(SNR[ri,:]-nfloor)/(nfloor+1e-6)
    if False:
        plt.pcolormesh(SNR,vmin=0,vmax=10)
        plt.colorbar()
        plt.show()
    sm=n.max(SNR)
#    print(sm)
    while sm > snr_min:
        mi=n.argmax(SNR)#.flatten())
#        print(mi)
        sm=n.max(SNR)
 #       print(sm)
        x,y=n.unravel_index(mi,SNR.shape)
     #   print(x,y)
        CS[x,y]+=sm
        ub=n.min([CS.shape[0],x+clen])
        lb=n.max([0,x-clen])
  #      print(SNR.shape)
        SNR[x,y]=0
        SNR[lb:ub,y]-=sm*0.15
    return(CS)
    if False:
        plt.pcolormesh(CS,vmin=0,vmax=10)
        plt.colorbar()
        plt.title("CLEAN")
        plt.show()

def rti_coh_incoh(fname="/data3/SLICE_RAW/MRRAW_20171220/raw.sodankyla.5a39ae21",
                  t_mess=0.06,
                  alias_num=0.0,
                  at0=None,
                  at1=None,
                  max_vel=1000,
                  mean_rem=False,
                  rem_iono=True,
                  n_incoh=10,
                  clean_range=False,
                  ofname="out.h5"):

    dname=re.search("(.*)/raw*",fname).group(1)
    os.system("mkdir -p %s/snr"%(dname))
    dname="%s/snr"%(dname)
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
    wf=ss.hann(n_time)
    freqs=n.fft.fftshift(n.fft.fftfreq(d.shape[2],d=r.IPP*r.n_integrations))
    vels=freqs*c.c/2.0/(r.frequency*1e6)
    bidx=n.where(n.abs(vels)>max_vel)[0]

    ipp_range=alias_num*r.IPP*c.c/2.0/1e3
    
    ranges=ipp_range + r.deltaR*n.arange(d.shape[1])+r.range_offset
    
    t0=r.time
    eff_prf=r.PRF/r.n_integrations
    dtau=(1.0/eff_prf)*d.shape[2]
    r=read_ud3.UD3_Reader(fname,t_mess=t_mess)

    RTI=[]
    DOP=[]
    tvec=[]
    n_avg=0
    S=n.zeros([d.shape[1],d.shape[2]],dtype=n.float32)
    for k in range(r.n_chunks):
        
        print("%d/%d"%(k,r.n_chunks))
        d=r.get_next_chunk()
        chunk_t0=r.time+dtau*k
        
        if chunk_t0 > at0:
            
            
            if rem_iono:
                chunk_t=n.mod(n.linspace(0,dtau,num=d.shape[2])+chunk_t0,60.0)
                d=remove_ionosonde(d,chunk_t)
            for i in range(n_chan):
                if i == 0 and False:
                    plt.pcolormesh(chunk_t,n.arange(d.shape[1]),n.abs(d[i,:,:]),vmax=1024)
                    plt.title(stuffr.unix2datestr(r.time + dtau*k))
                    plt.colorbar()
                    plt.show()

                for j in range(n_range):
                    z=d[i,j,:]
                    if mean_rem:
                        z=z-n.mean(z)
                    S[j,:]+=n.fft.fftshift(n.abs(n.fft.fft(wf*z))**2.0)
            if clean_range:
                S=clean_rti(S)
            else:
                S=snr_0(S)

    tvec=n.array(tvec)
    
    ho=h5py.File(ofname,"w")
    ho["tvec"]=tvec
    ho["ranges"]=ranges
    ho["RTI"]=RTI
    ho["DOP"]=DOP
    ho.close()
    
    return(tvec,ranges,RTI,DOP)

if __name__ == "__main__":
    if False:
#        tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/raw.andenes.20201204_132547.ud3",
 #                                         alias_num=1,
  #                                        at0=8600+1.60708e9,
   #                                       at1=8680+1.60708e9,                                      
    #                                      t_mess=0.2,
     #                                     mean_rem=True,
      #                                    ofname="and-trail-0.2s.h5")
        tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/raw.andenes.20201204_132547.ud3",
                                          alias_num=1,
                                          at0=8600+1.60708e9,
                                          at1=9000+1.60708e9,                                      
                                          t_mess=1.0,
                                          max_vel=150.0,
                                          mean_rem=True,
                                          clean_range=True,
                                          ofname="and-trail-1s.h5")
#    tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
                                      

   # tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/mr_trails/raw.sodankyla.5fc9d2a8",
  #                                    t_mess=0.06,
##                                      max_n_chunks=None)
    #tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
     #                                 alias_num=2,
      #                                at0=8600+1.60708e9,
       #                               at1=9050+1.60708e9,                                      
        #                              t_mess=1.0,
         #                             ofname="sod-trail-1s.h5")
    if True:
        tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
                                          alias_num=2,
                                          at0=8600+1.60708e9,
                                          at1=9050+1.60708e9,                                      
                                          t_mess=0.2,
                                          mean_rem=False,
                                          clean_range=False,
                                          ofname="sod-trail-0.2s.h5")
#    tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
 #                                     alias_num=2,
  #                                    at0=8600+1.60708e9,
   #                                   at1=9050+1.60708e9,                                      
    #                                  t_mess=0.06,
     #                                 ofname="sod-trail-0.06s.h5")
#    tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
 #   /media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/raw.andenes.20201204_132547.ud3

#    tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
 #                                     alias_num=2,
  #                                    at0=34+1.6070886e9,
   #                                   at1=48+1.6070886e9,
    #                                  t_mess=0.06,
     #                                 ofname="out-0.06s.h5")

    
    print(RTI.shape)
    dB=n.transpose(10.0*n.log10(RTI))
    nfloor=n.nanmedian(dB)
    plt.pcolormesh(tvec,ranges,dB,cmap="plasma",vmin=nfloor,vmax=nfloor+20.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Range (km)")
    plt.colorbar()
    plt.show()
    plt.pcolormesh(tvec,ranges,n.transpose(DOP),cmap="seismic",vmin=-100,vmax=100.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Range (km)")    
    plt.colorbar()
    plt.show()
