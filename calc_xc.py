#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import read_ud3
import h5py
import itertools
import stuffr
import glob
import scipy.signal as ss


def remove_ionosonde(d):
    for i in range(d.shape[0]):
        med_pwr=n.median(n.abs(d[i,:,:]),axis=0)
        avg_noise=n.median(med_pwr)
        std_est=n.median(n.abs(med_pwr-avg_noise))
        bad_idx=n.where(med_pwr-avg_noise > 5.0*std_est)[0]
        d[i,:,bad_idx]=0.0
    return(d)

def analyze_file(fname="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
                 odir="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/xc4",
                 remove_ionosonde=True,
                 t_mess=0.25):

    os.system("mkdir -p %s"%(odir))
    r=read_ud3.UD3_Reader(fname,t_mess=t_mess)
    n_int=40
    d=r.get_next_chunk()
    
    if remove_ionosonde:
        d=remove_ionosonde(d)
        
    n_chan=d.shape[0]
    n_range=d.shape[1]
    n_time=d.shape[2]
    
    wf=ss.hann(n_time)
    
    freqs=n.fft.fftshift(n.fft.fftfreq(d.shape[2],d=r.IPP*r.n_integrations))
    vels=freqs*3e8/2.0/36.9e6
    ranges=r.deltaR*n.arange(d.shape[1])+r.range_offset
    
    t0=r.time
    eff_ipp=r.PRF/r.n_integrations
    dtau=(1.0/eff_ipp)*d.shape[2]

    n_pairs=int(d.shape[0]*(d.shape[0]-1)/2)
    ant_pairs=list(itertools.combinations(n.arange(d.shape[0]),2))
    ant_pairs_m=n.array(ant_pairs)
    
    r=read_ud3.UD3_Reader(fname,t_mess=t_mess)
    
    for k in range(int(n.floor(r.n_chunks/n_int))):
        X=n.zeros([n_pairs,d.shape[1],d.shape[2]],dtype=n.complex64)
        S=n.zeros([d.shape[1],d.shape[2]])    
        for ti in range(n_int):
            d=r.get_next_chunk()
            if remove_ionosonde:
                d=remove_ionosonde(d)
            
            for j in range(n_range):
                for pi in range(n_pairs):
                    i0=ant_pairs[pi][0]
                    i1=ant_pairs[pi][1]

                    A=n.fft.fft(wf*d[i0,j,:])
                    B=n.fft.fft(wf*d[i1,j,:])
                    XS=n.fft.fftshift(A*n.conj(B))
                    X[pi,j,:]+=XS
                    S[j,:]+=n.abs(n.fft.fftshift(A*n.conj(B)))
        X=X/float(n_int)
        dB=10.0*n.log10(S)
        vmed=n.nanmedian(dB)
        dB=dB-vmed
        
        plt.clf()
        plt.pcolormesh(vels,ranges,dB,vmin=0,vmax=17,cmap="viridis")
        plt.colorbar()
        plt.xlabel("Doppler (m/s)")
        plt.ylabel("Range (km)")
        
        tdata=dtau*k*n_int + t0
        
        plt.title(stuffr.unix2datestr(tdata))
        plt.xlim([n.min(vels),n.max(vels)])
        plt.ylim([n.min(ranges),n.max(ranges)])
        
        plt.title(stuffr.unix2datestr(tdata))
        plt.savefig("%s/rd-%d.png"%(odir,tdata))
        plt.close()
        
        for pi in range(n_pairs):
            plt.clf()
            plt.pcolormesh(vels,ranges,n.angle(X[pi,:,:]),cmap="jet",vmin=-n.pi,vmax=n.pi)
            plt.title(ant_pairs[pi])
            plt.colorbar()
            plt.xlabel("Doppler (m/s)")
            plt.ylabel("Range (km)")
            plt.xlim(-200,200)
            plt.title("%s (%d,%d)"%(stuffr.unix2datestr(tdata),ant_pairs[pi][0],ant_pairs[pi][1]))
            plt.ylim([n.min(ranges),n.max(ranges)])
            plt.savefig("%s/an-%d-%03d.png"%(odir,tdata,pi))
            plt.close()
            
            plt.clf()
            coh=n.abs(X[pi,:,:])
            plt.pcolormesh(vels,ranges,coh,cmap="viridis",vmin=0,vmax=1.0)
            plt.colorbar()
            plt.xlabel("Doppler (m/s)")
            plt.ylabel("Range (km)")
            plt.xlim(-250,250)
            
            plt.ylim([n.min(ranges),n.max(ranges)])
            
            plt.title(ant_pairs[pi])
            plt.savefig("%s/xc-%d-%03d.png"%(odir,tdata,pi))
            plt.close()
            

        ho=h5py.File("%s/rd-%d.h5"%(odir,tdata),"w")
        ho["t0"]=tdata
        ho["X"]=X
        ho["S"]=S        
        ho["ant_pairs"]=ant_pairs_m
        ho["range"]=ranges
        ho["vels"]=vels
        ho.close()

analyze_file()

    
