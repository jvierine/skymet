#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import read_ud3
import h5py
import itertools
import stuffr
import glob
import scipy.signal as ss
import scipy.constants as c
import os
import scipy.interpolate as si

def remove_ionosonde(d,
                     t,
                     tbad=[0.8,1.525,2.35,3.4,4.68,6.45,8.9,12.6],
                     ts=0.1):
    w = n.zeros(len(t),dtype=n.float32)
    for t0 in tbad:
        w+=n.exp(-(t-t0)**2.0/(2.0*ts**2.0))
#    plt.plot(1-w)
 #   plt.show()
    for ti in range(d.shape[2]):
        d[:,:,ti]=d[:,:,ti]*(1.0-w[ti])
    return(d)


def analyze_file(fname="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/mr_trails/raw.sodankyla.5fc9d2a8",
                 odir="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/mr_trails/xc2",
                 rem_ionosonde=True,
                 plot_coherence=False,
                 plot_snr=True,
                 rem_mean=False,
                 n_int=20,
                 plot_angles=False,
                 t_mess=0.25):
    """
    t_mess = coherent integration length (seconds)
    rem_ionosonde = perform outlier remove to reduce ionosonde related RFI
    fname = the raw data file from a skymet radar
    odir = the output directory.
    n_int = number of incoherent integrations (time step is t_mess*n_int)
    """
    os.system("mkdir -p %s"%(odir))
    os.system("rm %s/*"%(odir))    
    r=read_ud3.UD3_Reader(fname,t_mess=t_mess)
    d=r.get_next_chunk()
    n_chan=d.shape[0]
    n_range=d.shape[1]
    n_time=d.shape[2]
    
    wf=ss.hann(n_time)
    
    freqs=n.fft.fftshift(n.fft.fftfreq(d.shape[2],d=r.IPP*r.n_integrations))
    vels=freqs*c.c/2.0/(1e6*r.frequency)
    ranges=(r.deltaR)*n.arange(n_range)+r.range_offset
    ipp_range=r.IPP*c.c/2.0/1e3
    
    t0=r.time
    eff_prf=r.PRF/r.n_integrations
    dtau=(1.0/eff_prf)*d.shape[2]

    n_pairs=int(d.shape[0]*(d.shape[0]-1)/2)
    ant_pairs=list(itertools.combinations(n.arange(d.shape[0]),2))
    ant_pairs_m=n.array(ant_pairs)

    antenna_coords=r.antenna_coords
    
    r=read_ud3.UD3_Reader(fname,t_mess=t_mess)

    n_files = int(n.floor(r.n_chunks/n_int))
    for k in range(n_files):
        print("%d/%d"%(k,n_files))
        X=n.zeros([n_pairs,n_range,d.shape[2]],dtype=n.complex64)
        S=n.zeros([n_range,d.shape[2]])
        ws=0.0
        for ti in range(n_int):
            d=r.get_next_chunk()
            if rem_ionosonde:
                chunk_t0=r.time+dtau*ti
                chunk_t=n.mod(n.linspace(0,dtau,num=d.shape[2])+chunk_t0,60.0)
                d=remove_ionosonde(d,chunk_t)                

            for j in range(n_range):
                for pi in range(n_pairs):
                    i0=ant_pairs[pi][0]
                    i1=ant_pairs[pi][1]
                    za=d[i0,j,:]
                    zb=d[i1,j,:]
                    if rem_mean:
                        za=za-n.mean(za)
                        zb=zb-n.mean(zb)
                    A=n.fft.fft(wf*za)
                    B=n.fft.fft(wf*zb)
                    XS=n.fft.fftshift(A*n.conj(B))
                    axc=n.abs(XS)
                    w=n.median(axc)
                    if w > 1:
                        X[pi,j,:]+=(1.0/w)*XS
                        S[j,:]+=(1.0/w)*axc
                        ws+=1.0/w
        X=X/ws
        S=S/ws
        dB=10.0*n.log10(S)
        vmed=n.nanmedian(dB)
        dB=dB-vmed
        tdata=dtau*k*n_int + t0


        if plot_snr:
            plt.pcolormesh(vels,ranges,dB,vmin=0,vmax=17,cmap="viridis")
            plt.colorbar()
            plt.xlabel("Doppler (m/s)")
            plt.ylabel("Range (km)")
            plt.title(stuffr.unix2datestr(tdata))
            plt.xlim([n.min(vels),n.max(vels)])
            plt.ylim([n.min(ranges),n.max(ranges)])
            plt.title(stuffr.unix2datestr(tdata))
            plt.savefig("%s/rd-%d.png"%(odir,tdata))
            plt.close()
            plt.clf()
            
        for pi in range(n_pairs):
            if plot_angles:
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
            
            if plot_coherence:
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
                plt.clf()

        ho=h5py.File("%s/rd-%d.h5"%(odir,tdata),"w")
        ho["t0"]=tdata
        ho["X"]=X
        ho["S"]=S        
        ho["ant_pairs"]=ant_pairs_m
        ho["range"]=ranges
        ho["ipp_range"]=ipp_range
        ho["vels"]=vels
        ho["antenna_coords"]=antenna_coords
        ho.close()


analyze_file(fname="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
             odir="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/xc6",
             rem_ionosonde=True,
             plot_coherence=False,
             plot_snr=True,
             plot_angles=False,
             t_mess=0.5)
#analyze_file(fname="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/raw.andenes.20201204_132547.ud3",
#             odir="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/and_mr/xc",
#             rem_ionosonde=False,
#             plot_coherence=False,
#             plot_snr=True,
#             rem_mean=False,
#             plot_angles=False,
#             t_mess=0.25)
        
#analyze_file()

    
