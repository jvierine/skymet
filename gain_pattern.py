import numpy as n
import scipy.interpolate as sio
import matplotlib.pyplot as plt

def skymet_gain(el):
    el_angs=90.0-n.array([90, 85, 80, 70,  60,  50,  40, 30,   20,  10,  0])
    gains=n.array([      -25,-11, -5,  -1,  2,  4,    5,  5.1, 5.3, 5.4, 5.5])
    gf=sio.interp1d(el_angs,gains,kind="quadratic")
#    pc=n.polyfit(el_angs,gains,3)
#    gf=n.poly1d(pc)
    return(gf(n.abs(el)))

if __name__  == "__main__":
    els=n.linspace(0,90,num=100)
    
    plt.plot(els,skymet_gain(els))
    plt.show()
    
