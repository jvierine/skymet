import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c


def rcs_est(G=2.0,
            P_tx=30*4*15e3,
            lam=c.c/36.9e6,
            T_sys=9000,
            B=50e3,
            R=170e3,
            SNR=1e3):
    return(SNR*c.k*T_sys*B*(4.0*n.pi)**3.0*R**4/(P_tx*G**2.0*lam**2.0))


if __name__ == "__main__":
    print(rcs_est())
    
    
            
