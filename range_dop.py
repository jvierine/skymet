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


def rti_coh_incoh(fname="/data3/SLICE_RAW/MRRAW_20171220/raw.sodankyla.5a39ae21",
                  t_mess=0.06,
                  alias_num=0.0,
                  at0=None,
                  at1=None,
                  ofname="out.h5"):


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
    eff_prf=r.PRF/r.n_integrations
    print(eff_prf)
    dtau=(1.0/eff_prf)*d.shape[2]
    r=read_ud3.UD3_Reader(fname,t_mess=t_mess)
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
    for k in range(n_c):
        print("%d/%d"%(k,n_c))
        S=n.zeros([d.shape[1],d.shape[2]])
        d=r.get_next_chunk()
        print(stuffr.unix2datestr(r.time + dtau*k))
#        print(d)
#        if d == None:
 #           break
        if k in gidx:
            print(d.shape)
            tvec.append(r.time + dtau*k)
            for i in range(n_chan):
                if i == 0 and False:
                    plt.pcolormesh(n.abs(d[i,:,:]),vmin=0,vmax=512)
                    plt.title(stuffr.unix2datestr(r.time + dtau*k))
                    plt.colorbar()
                    plt.show()

                for j in range(n_range):
                    z=d[i,j,:]
                    z=z-n.median(z)
                    S[j,:]+=n.fft.fftshift(n.abs(n.fft.fft(z))**2.0)
            for j in range(n_range):
                RTI[oidx,j]=n.max(S[j,:])
                DOP[oidx,j]=vels[n.argmax(S[j,:])]
            oidx+=1
    tvec=n.array(tvec)
    
    ho=h5py.File(ofname,"w")
    ho["tvec"]=tvec
    ho["ranges"]=ranges
    ho["RTI"]=RTI
    ho["DOP"]=DOP
    ho.close()
    
    return(tvec,ranges,RTI,DOP)

if __name__ == "__main__":
    tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/raw.andenes.20201204_132547.ud3",
                                      alias_num=0,
                                      at0=8600+1.60708e9,
                                      at1=9040+1.60708e9,                                      
                                      t_mess=10.0,
                                      ofname="and-trail-10s.h5")
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
#    tvec,ranges,RTI,DOP=rti_coh_incoh("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
 #                                     alias_num=2,
  #                                    at0=8600+1.60708e9,
   #                                   at1=9050+1.60708e9,                                      
    #                                  t_mess=10.0,
     #                                 ofname="sod-trail-10s.h5")
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
    plt.pcolormesh(tvec,ranges,dB,cmap="plasma",vmin=nfloor,vmax=nfloor+30.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Range (km)")
    plt.colorbar()
    plt.show()
    plt.pcolormesh(tvec,ranges,n.transpose(DOP),cmap="seismic",vmin=-100,vmax=100.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Range (km)")    
    plt.colorbar()
    plt.show()
