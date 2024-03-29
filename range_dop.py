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

def remove_ionosonde(d):
    for i in range(d.shape[0]):
        med_pwr=n.median(n.abs(d[i,:,:]),axis=0)
        avg_noise=n.median(med_pwr)
        std_est=n.median(n.abs(med_pwr-avg_noise))
        bad_idx=n.where(med_pwr-avg_noise > 5.0*std_est)[0]
        d[i,:,bad_idx]=0.0
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
    # mono-static Doppler
    # v = c*df/2.0/f
    vels=freqs*c.c/2.0/(r.frequency*1e6)
    bidx=n.where(n.abs(vels)>max_vel)[0]

    ipp_range=alias_num*r.IPP*c.c/2.0/1e3
    
    ranges=ipp_range + r.deltaR*n.arange(d.shape[1])+r.range_offset
    
    t0=r.time
    eff_prf=r.PRF/r.n_integrations
    print(eff_prf)
    dtau=(1.0/eff_prf)*d.shape[2]
    r=read_ud3.UD3_Reader(fname,t_mess=t_mess)
    shape=n.copy(d.shape)
    print(d.shape)

    n_c=r.n_chunks-1

    tvec_all=n.arange(n_c)*dtau + r.time
    if t0 != None:
        gidx=n.where( (tvec_all > at0) & (tvec_all < at1))[0]
    else:
        gidx=n.arange(n_c)
    n_tv=len(gidx)

        
    RTI=n.zeros([n_tv,n_range],dtype=n.float32)
    DOP=n.zeros([n_tv,n_range],dtype=n.float32)
        
    tvec=[]
    oidx=0
    noises=[]
    for k in range(n_c):
        print("%d/%d"%(k,n_c))
        d=r.get_next_chunk()
        
        if k in gidx:
            S=n.zeros([d.shape[1],d.shape[2]])
            R=n.zeros([d.shape[1],d.shape[2]])
            noise=n.zeros(d.shape[1])
            print(stuffr.unix2datestr(r.time + dtau*k))
            tvec.append(r.time + dtau*k)
            
            for i in range(n_chan):                
                #                if i == 0 and True:
                R+=n.abs(d[i,:,:])**2.0

                for j in range(n_range):
                    
                    z=d[i,j,:]
                    
                    if mean_rem:
                        z=z-n.mean(z)
                        
                    S[j,:] += n.fft.fftshift(n.abs(n.fft.fft(wf*z))**2.0)
            dBR=10.0*n.log10(R)
            nf=n.nanmedian(dBR)
            plt.pcolormesh(dBR,vmin=nf,vmax=nf+20)
            plt.title(stuffr.unix2datestr(r.time + dtau*k))
            plt.colorbar()
            plt.show()

            if clean_range:
                S=clean_rti(S)
            else:
                S=snr_0(S)
            for j in range(n_range):
                S[j,bidx]=0.0                
                noise[j]=n.median(S[j,:])
                RTI[oidx,j]=n.max(S[j,:])
                DOP[oidx,j]=vels[n.argmax(S[j,:])]
                
            hos=h5py.File("%s/snr-%d.h5"%(dname,r.time+dtau*k),"w")
            hos["t0"]=r.time+dtau*k
            hos["ranges"]=ranges
            hos["vels"]=vels
            hos["S"]=S
            hos.close()
            noises.append(noise)   
            if False:
                S[S<=0]=1e-3
                plt.pcolormesh(vels,ranges,10.0*n.log10(S),vmin=0,vmax=40)
                plt.title(stuffr.unix2datestr(r.time+dtau*k))
                plt.xlim([-200,200])
                plt.colorbar()
                plt.tight_layout()
                plt.savefig("%s/snr-%d.png"%(dname,r.time+dtau*k))
                plt.clf()
                plt.close()
                
            oidx+=1
    tvec=n.array(tvec)
    
    ho=h5py.File(ofname,"w")
    ho["tvec"]=tvec
    ho["ranges"]=ranges
    ho["RTI"]=RTI
    ho["DOP"]=DOP
    ho["noise"]=n.array(noises)
    ho["shape"]=shape
    ho.close()
    
    return(tvec,ranges,RTI,DOP)

if __name__ == "__main__":
    head=True
    if head:
        tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/raw.andenes.20201204_132547.ud3",
                                          alias_num=1,
                                          at0=8600+1.60708e9+35,
                                          at1=8680+1.60708e9,                                      
                                          t_mess=0.06,
                                          mean_rem=True,
                                          ofname="and-trail-0.06s.h5")
        
        tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
                                          alias_num=2,
                                          at0=34+1.6070886e9,
                                          at1=48+1.6070886e9,
                                          t_mess=0.06,
                                          ofname="out-0.06s.h5")

    
    if False:
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
    if False:
        tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
                                          alias_num=2,
                                          at0=8600+1.60708e9,
                                          at1=9050+1.60708e9,                                      
                                          t_mess=3.0,
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

 

    
