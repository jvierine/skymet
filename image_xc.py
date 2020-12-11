#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
#plt.style.use("dark_background")
import h5py
import glob
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import scipy.constants as c

#import aoa
import stuffr
import h5py
import jcoord
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD    

class aoa_find:
    def __init__(self,n_d,u,v,f=36.9e6,maxr=1.0,psf_w=5):
        self.lam=c.c/f
        self.u=u
        self.v=v
        
        self.dcoss_x=n.linspace(-1,1,num=n_d)
        self.dcoss_y=n.linspace(-1,1,num=n_d)
        self.dsx,self.dsy=n.meshgrid(self.dcoss_x,self.dcoss_y)
        self.bad_idx=n.where(n.sqrt(self.dsx**2.0+self.dsy**2.0)>maxr)
        self.phase_diffs=[]

        self.az=n.zeros(self.dsx.shape)
        self.el=n.zeros(self.dsx.shape)        

        for xi in range(self.dsx.shape[0]):
            for yi in range(self.dsx.shape[1]):
                l_h = n.sqrt(self.dsx[xi,yi]**2.0 + self.dsy[xi,yi]**2.0)
                if l_h <= 1.0:
                    self.az[xi,yi]=180.0*n.arctan2(self.dsx[xi,yi],self.dsy[xi,yi])/n.pi
                    l_v = n.sqrt(1.0-l_h**2.0)
                    self.el[xi,yi]=180.0*n.arctan(l_v/l_h)/n.pi
                
        for i in range(len(u)):
            self.phase_diffs.append(2.0*n.pi*self.dsx*self.u[i]/self.lam+2.0*n.pi*self.dsy*self.v[i]/self.lam)
                        


def find_ds(m_phase_diffs,a):
    """
    find direction of arrival using a matched filter (beamforming)
    """
    pds=n.zeros(a.dsx.shape,dtype=n.complex64)
    pds[:,:]=0.0

    for txi in range(len(a.u)):
        pds+=n.exp(1j*a.phase_diffs[txi])*n.conj(m_phase_diffs[txi])

    pds[a.bad_idx]=0.0
    mi=n.argmax(n.abs(pds)**2.0)
    x,y=n.unravel_index(mi,pds.shape)
    
    plott=False
    if plott:
        plt.pcolormesh(n.abs(pds))
        plt.plot(x,y,"x",color="white")
        plt.xlabel("Direction cosine (x)")
        plt.ylabel("Direction cosine (y)")        
        plt.colorbar()
        plt.show()
    I=n.zeros(pds.shape)
    I[x,y]=1.0
    return(a.dcoss_x[x],a.dcoss_y[y],a.az[x,y],a.el[x,y],I)

#68.25407756526351,
#24.23106261125355,
#300000,
#=300000,
def xc2img(fname="r1s/rd-1529980785.h5",
           odir="./r1s/",
           obs_lat=67.365524,
           obs_lon=26.637495,
           map_lat=68.0,
           map_lon=26.0,
           obs_h=500.0,
           dB_thresh=5.0,
           alias_num=1.0,
           plot_map=True,
           plot_latlongh=True,
           plot_direction_cosines=True,
           map_width=300e3,
           map_height=300e3,
           gc_len=5,
           f_smooth=3,
           height_color=True,
           N=400,
           t0=0.0):
    
    h=h5py.File(fname,"r")
    if h["t0"].value < t0:
        h.close()
        return

    range_gates=h["range"].value + h["ipp_range"].value*alias_num

    # antenna positions: range from array center
    a_range=h["antenna_coords"].value[:,0]
    # antenna positions: azimuth relative to array center
    a_az=h["antenna_coords"].value[:,1]

    # x and y positions in local horizontal plane with x east-west and y north-south
    rx_y=-a_range*n.cos(n.pi*a_az/180.0)
    rx_x=-a_range*n.sin(n.pi*a_az/180.0)

    # antenna index pairs
    aidx=n.copy(h["ant_pairs"].value)

    # cross-correlation-range-doppler data cube
    X=n.copy(h["X"].value)
    if f_smooth > 1:
        for ai in range(X.shape[0]):
            for ri in range(X.shape[1]):
                row=n.convolve(n.repeat(1.0/f_smooth,f_smooth),X[ai,ri,:],mode="same")
                X[ai,ri,:]=row

    # calculate antenna position vectors
    u=[]
    v=[]
    for i in range(aidx.shape[0]):
        u.append(rx_x[aidx[i,0]]-rx_x[aidx[i,1]])
        v.append(rx_y[aidx[i,0]]-rx_y[aidx[i,1]])
    u=n.array(u)
    v=n.array(v)

    # snr
    S=h["S"].value
    # doppler velocities
    vels=h["vels"].value

    # snr in dB
    dB=10.0*n.log10(S)
    noise_floor=n.nanmedian(dB)
    dB=dB-noise_floor
    dB[0:gc_len,:]=0.0
    for ri in range(dB.shape[0]):
        dB[ri,:]=dB[ri,:]-n.median(dB[ri,:])

#    plt.pcolormesh(dB)
#    plt.colorbar()
 #   plt.show()
    # find pixels to image where snr is sufficiently high
    gidx=n.where(dB.flatten() > dB_thresh)[0]
    gidx=n.unravel_index(gidx,dB.shape)

    # number of direction cosines to sample in the east and west direction
    # N^2 directions in search space
    a=aoa_find(N,u,v)
                        
    I=n.zeros([N,N],dtype=n.float32)
    V=n.zeros([N,N],dtype=n.float32)
    Nm=n.zeros([N,N])
    dcx=[]
    dcy=[]
    pv=[]
    plats=[]
    plons=[]
    phgts=[]        
    pp=[]            
    for i in range(len(gidx[0])):
        xi=gidx[0][i]
        yi=gidx[1][i]    
    
        print("%s %d/%d dB %1.2f"%(stuffr.unix2datestr(h["t0"].value),i,len(gidx[0]),dB[xi,yi]))

        m_phase_diffs=n.exp(1j*n.angle(X[:,xi,yi]))
        dcosx,dcosy,az,el,I0=find_ds(m_phase_diffs,a)
        r0 = range_gates[xi]*1e3

        # translate to lat long height
        llh=jcoord.az_el_r2geodetic(obs_lat, obs_lon, 0.0, az, el, r0)
        plats.append(llh[0])
        plons.append(llh[1])
        phgts.append(llh[2])        
        
        dcx.append(dcosx)
        dcy.append(dcosy)
        pv.append(h["vels"].value[yi])
        pp.append(dB[xi,yi])
        
        I+=I0*S[xi,yi]
        V+=I0*h["vels"].value[yi]
        Nm+=I0
    nidx=n.where(Nm>0)
    V[nidx]=V[nidx]/Nm[nidx]
    I[nidx]=I[nidx]/Nm[nidx]
    dcx=n.array(dcx)
    dcy=n.array(dcy)

    plons=n.array(plons)
    plats=n.array(plats)
    phgts=n.array(phgts)    
    pp=n.array(pp)
    pv=n.array(pv)
    
    ecefs=jcoord.geodetic2ecef(plats, plons, phgts)
    ho=h5py.File("%s/points-%d.h5"%(odir,h["t0"].value),"w")
    ho["ecef_m"]=ecefs
    ho["lon_deg"]=plons
    ho["lat_deg"]=plats
    ho["height_km"]=phgts/1e3
    ho["SNR_dB"]=pp
    ho["dop_vel"]=pv
    ho["t0"]=h["t0"].value
    ho.close()

    if plot_map:
        # setup Lambert Conformal basemap.False68.29911309985064, 23.3381106574698
        m = Basemap(width=map_width,
                    height=map_height,
                    projection='lcc',
                    resolution='i',
                    lat_0=map_lat,
                    lon_0=map_lon)
        # draw coastlines.
        m.drawmapboundary(fill_color="black")
        try:
            m.drawcoastlines(color="white")
        except:
            print("no coastlines")
        m.drawcountries(color="white")
        parallels = n.arange(0.,81,1.)
        # labels = [left,right,top,bottom]
        m.drawparallels(parallels,labels=[True,False,False,False],color="white")
        meridians = n.arange(10.,351.,2.)
        m.drawmeridians(meridians,labels=[False,False,False,True],color="white")
        print("got here")
        oidx=n.argsort(pp)
        if len(pp)>0:
            xlat,ylon=m(plons,plats)
            
            #            
            if height_color:
                m.scatter(xlat[oidx],ylon[oidx],c=phgts[oidx]/1e3,s=1.0,marker="o",vmin=80,vmax=105)
                cb=plt.colorbar()
                cb.set_label("Height (km)")
            else:
                m.scatter(xlat[oidx],ylon[oidx],c=pp[oidx],s=1.0,marker="o",vmin=dB_thresh,vmax=40.0)
                cb=plt.colorbar()
                cb.set_label("SNR (dB)")                
                
            
        plt.title(stuffr.unix2datestr(h["t0"].value))
        plt.tight_layout()
        plt.savefig("%s/map-%d.png"%(odir,h["t0"].value))
        plt.close()
        plt.clf()

        if len(plats)> 0 and plot_latlongh:
            ecefs=jcoord.geodetic2ecef(plats, plons, phgts)
            plt.subplot(121)
            plt.scatter(plats[oidx],phgts[oidx]/1e3,c=pp[oidx],vmin=0,vmax=40.0)
            plt.colorbar()
            plt.xlabel("Latitude (deg)")
            plt.ylabel("Height (km)")
            plt.xlim([67.5,68.2])
            plt.ylim([80,110.0]) 
            plt.subplot(122)
            plt.scatter(plons[oidx],phgts[oidx]/1e3,c=pp[oidx],vmin=0,vmax=40.0)
            plt.colorbar()
            plt.xlim([24.5,26.])
            plt.ylim([80,110.0]) 
            plt.xlabel("Longitude (deg)")
            plt.ylabel("Height (km)")
            plt.tight_layout()
            plt.savefig("%s/points-%d.png"%(odir,h["t0"].value))
            plt.close()

        
    if plot_direction_cosines:
        oidx=n.argsort(pp)
        fig=plt.figure()
#        plt.style.use('dark_background')
        phi=n.linspace(0,2*n.pi,num=500)
        plt.scatter(-dcx[oidx],-dcy[oidx],c=pv[oidx],vmin=-50,vmax=50,edgecolors="none",cmap="Spectral")
        plt.colorbar()    
        plt.plot(n.cos(phi),n.sin(phi),color="black")
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])    
        plt.title(stuffr.unix2datestr(h["t0"].value))
        
        plt.savefig("%s/vel-%d.png"%(odir,h["t0"].value))
        plt.close()

        fig=plt.figure()
 #       plt.style.use('dark_background')
        
        plt.scatter(-dcx[oidx],-dcy[oidx],c=pp[oidx],vmin=0,vmax=40,edgecolors="none")
        plt.colorbar()    
        plt.plot(n.cos(phi),n.sin(phi),color="black")    
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])    
        plt.title(stuffr.unix2datestr(h["t0"].value))    
        plt.savefig("%s/img-%d.png"%(odir,h["t0"].value))
        plt.close()

    h.close()

def image_files(dirname="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/mr_trails/xc"):
    """
    Go through all averaged cross-correlation-range-doppler files.
    """
    fl=glob.glob("%s/rd*.h5"%(dirname))
    fl.sort()
    for fi in range(comm.rank,len(fl),comm.size):
        f=fl[fi]
        xc2img(fname=f,odir=dirname)
        
if __name__ == "__main2__":

    dirname="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/mr_trails/xc"
    map_lat=68.25407756526351
    map_lon=24.23106261125355
    map_width=300e3
    map_height=300e3
    t0=1607088618.0
    fl=glob.glob("%s/rd*.h5"%(dirname))
    fl.sort()
    for fi in range(comm.rank,len(fl),comm.size):
        f=fl[fi]
        xc2img(fname=f,
               odir=dirname,
               obs_lat=67.365524,
               obs_lon=26.637495,
               map_lat=map_lat,
               map_lon=map_lon,
               obs_h=500.0,
               dB_thresh=5.0,
               alias_num=2.0,
               plot_map=True,
               plot_latlongh=True,
               plot_direction_cosines=True,
               map_width=map_width,
               map_height=map_height,
               gc_len=5,
               f_smooth=3,
               N=400,
               t0=t0)
if __name__ == "__main__":
#    image_files(dirname="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/mr_trails/xc")
    dirname="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/mr_trails/xc"
    map_lat=67.75
    map_lon=25.0
    map_width=150e3
    map_height=150e3
    t0=1607062684.0
    fl=glob.glob("%s/rd*.h5"%(dirname))
    fl.sort()
    for fi in range(comm.rank,len(fl),comm.size):
        f=fl[fi]
        xc2img(fname=f,
               odir=dirname,
               obs_lat=67.365524,
               obs_lon=26.637495,
               map_lat=map_lat,
               map_lon=map_lon,
               obs_h=500.0,
               dB_thresh=5.0,
               alias_num=1.0,
               plot_map=True,
               plot_latlongh=True,
               plot_direction_cosines=True,
               map_width=map_width,
               map_height=map_height,
               gc_len=5,
               f_smooth=3,
               N=400,
               t0=t0)
