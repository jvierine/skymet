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
                        
# antenna coordinates
a_range=n.array([16.25, 20.31, 16.25, 20.31, 0.0, 40.63, 48.75])
a_az=n.array([0, 180, 90, 270, 0, 90, 180.0])
rx_y=-a_range*n.cos(n.pi*a_az/180.0)
rx_x=-a_range*n.sin(n.pi*a_az/180.0)

h=h5py.File("out_536.h5","r")
range_gates=n.copy(h["ranges"].value)
obs_lat=67.365524
obs_lon=26.637495
print(h.keys())
h.close()
def find_ds(m_phase_diffs,a):
    
    pds=n.zeros(a.dsx.shape,dtype=n.complex64)
    pds[:,:]=0.0

    for txi in range(len(a.u)):
        pds+=n.exp(1j*a.phase_diffs[txi])*n.conj(m_phase_diffs[txi])#n.exp(-1j*m_phase_diffs[txi])

    pds[a.bad_idx]=0.0
    mi=n.argmax(n.abs(pds)**2.0)
    x,y=n.unravel_index(mi,pds.shape)
    
    plott=False
    if plott:
#        model_phase_diffs=n.zeros(len(m_phase_diffs))
 #       for txi in range(len(u)):
  #          model_phase_diffs[txi]=2.0*n.pi*a.dsx[x,y]*a.u[txi]/a.lam+2.0*n.pi*a.dsy[x,y]*a.v[txi]/a.lam

   #     resid=n.exp(1j*m_phase_diffs)-n.exp(1j*model_phase_diffs)
    #    plt.plot(resid.real,"o")
     #   plt.plot(resid.imag,"o")        
      #  plt.show()
        
        plt.pcolormesh(n.abs(pds))
        plt.plot(x,y,"x",color="white")
        plt.xlabel("Direction cosine (x)")
        plt.ylabel("Direction cosine (y)")        
        plt.colorbar()
        plt.show()
    I=n.zeros(pds.shape)
    I[x,y]=1.0#n.abs(pds[x,y])#1.0
#    I=n.fft.ifft2(n.fft.fft2(I)*n.fft.fft2(a.psf,s=(I.shape[0],I.shape[1]))).real
    return(a.dcoss_x[x],a.dcoss_y[y],a.az[x,y],a.el[x,y],I)

def xc2img(fname="r1s/rd-1529980785.h5",
           odir="./r1s/",
           dB_thresh=10.0,
           fdec=4,
           t0=1607088600.0):
    
    h=h5py.File(fname,"r")
    if h["t0"].value < t0:
        h.close()
        return
    aidx=n.copy(h["ant_pairs"].value)
    X=n.copy(h["X"].value)

    u=[]
    v=[]
    for i in range(aidx.shape[0]):
        u.append(rx_x[aidx[i,0]]-rx_x[aidx[i,1]])
        v.append(rx_y[aidx[i,0]]-rx_y[aidx[i,1]])
    u=n.array(u)
    v=n.array(v)

    S=h["S"].value
    vels=h["vels"].value
    
    dB=10.0*n.log10(S)
    noise_floor=n.nanmedian(dB)
    dB=dB-noise_floor
    dB[0:5,:]=0.0
    dB[0:10,260:275]=0.0
    dB[:,188:194]=0.0

    # peak doppler only
#    for ri in range(dB.shape[0]):
 #       di=n.argmax(dB[ri,:])
  #      tmp=dB[ri,di]
   #     dB[ri,:]=0.0
    #    dB[ri,di]=tmp

  #  plt.pcolormesh(dB)
 #   plt.colorbar()
#    plt.show()
    gidx=n.where(dB.flatten() > dB_thresh)[0]
    gidx=n.unravel_index(gidx,dB.shape)
                                    
    N=400
    a=aoa_find(N,u,v)
                        
    I=n.zeros([N,N])
    V=n.zeros([N,N])
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
        print(gidx[0][i],gidx[1][i])
        m_phase_diffs=n.exp(1j*n.angle(X[:,xi,yi]))
#        dcosx,dcosy,I0=find_ds(X[:,xi,yi],a)
        dcosx,dcosy,az,el,I0=find_ds(m_phase_diffs,a)
        print(az)
        print(el)
        r0 = range_gates[xi]*1e3
        print(r0)
        llh=jcoord.az_el_r2geodetic(obs_lat, obs_lon, 0.0, az, el, r0)
        print(llh)
        plats.append(llh[0])
        plons.append(llh[1])
        phgts.append(llh[2])        
        
        dcx.append(dcosx)
        dcy.append(dcosy)
        pv.append(h["vels"].value[yi])
        pp.append(dB[xi,yi])
        I+=I0*S[xi,yi]
#        plt.pcolormesh(I)
 #       plt.colorbar()
  #      plt.show()
        V+=I0*h["vels"].value[yi]
        Nm+=I0
    nidx=n.where(Nm>0)
    V[nidx]=V[nidx]/Nm[nidx]
#    I[nidx]=I[nidx]#/Nm[nidx]
    dcx=n.array(dcx)
    dcy=n.array(dcy)

    plons=n.array(plons)
    plats=n.array(plats)
    phgts=n.array(phgts)    
    pp=n.array(pp)
    print(plons)
    print(plats)
    if True:
        # setup Lambert Conformal basemap.False68.29911309985064, 23.3381106574698
        m = Basemap(width=300000,height=300000,projection='lcc',
                    resolution='i',lat_0=69,lon_0=23.)
        xlat,ylon=m(plons,plats)
        # draw coastlines.
        m.drawmapboundary(fill_color="black")
        m.drawcoastlines(color="white")
        m.drawcountries(color="white")
        oidx=n.argsort(pp)
        m.scatter(xlat[oidx],ylon[oidx],c=pp[oidx],s=1.0,marker="o",vmin=dB_thresh,vmax=40.0)
        plt.title(stuffr.unix2datestr(h["t0"].value))
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar()
        # draw a boundary around the map, fill the background.
        # this background will end up being the ocean color, since
        # the continents will be drawn on top.
        #        m.drawmapboundary(fill_color='aqua')
        # fill continents, set lake color same as ocean color.
        #       m.fillcontinents(color='coral',lake_color='aqua')
        plt.savefig("%s/img-%d.png"%(odir,h["t0"].value))
        plt.close()
        plt.clf()

        if len(plats)> 0 and False:
            ecefs=jcoord.geodetic2ecef(plats, plons, phgts)
            plt.subplot(121)
            plt.scatter(plats[oidx],phgts[oidx]/1e3,c=pp[oidx],vmin=0,vmax=40.0)
            plt.colorbar()
            plt.xlabel("Latitude (deg)")
            plt.ylabel("Height (km)")            
            plt.subplot(122)
            plt.scatter(plons[oidx],phgts[oidx]/1e3,c=pp[oidx],vmin=0,vmax=40.0)
            plt.colorbar()
#            plt.scatter(plons,phgts/1e3,".")
            plt.xlabel("Longitude (deg)")
            plt.ylabel("Height (km)")            
            plt.show()
        
#        plt.show()

    if False:
        fig=plt.figure()
        plt.style.use('dark_background')
        phi=n.linspace(0,2*n.pi,num=500)
        plt.scatter(dcx,dcy,c=pv,vmin=-150,vmax=150,edgecolors="none")
        plt.colorbar()    
        plt.plot(n.cos(phi),n.sin(phi),color="white")
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])    
        plt.title(stuffr.unix2datestr(h["t0"].value))
        
        plt.savefig("%s/vel-%d.png"%(odir,h["t0"].value))
        plt.close()
        
        #    img=10.0*n.log10(n.transpose(I))-nfloor
        fig=plt.figure()
        plt.style.use('dark_background')
        
        plt.scatter(dcx,dcy,c=10.0*n.log10(pp),vmin=0,vmax=20,edgecolors="none")
        plt.colorbar()    
        plt.plot(n.cos(phi),n.sin(phi),color="white")    
        plt.xlim([-1.1,1.1])
        plt.ylim([-1.1,1.1])    
        #    plt.pcolormesh(img[:,::-1],vmin=0,vmax=20,cmap="viridis")
        plt.title(stuffr.unix2datestr(h["t0"].value))    
        
        plt.savefig("%s/img-%d.png"%(odir,h["t0"].value))
        plt.close()


    h.close()

#    fname="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/raw.sodankyla.5fca359e",
#odir="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/xc"):
fl=glob.glob("/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/xc3/*.h5")
fl.sort()

for fi in range(comm.rank,len(fl),comm.size):
    f=fl[fi]
    print(f)
    xc2img(fname=f,odir="/media/j/4f5bab17-2890-4bb0-aaa8-ea42d65fdac8/bolide_20201204/sod_mr/xc3")
