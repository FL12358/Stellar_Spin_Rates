import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits

def Reddening():
    kG = [0.9761, -0.1704, 0.0086, 0.0011, -0.0438, 0.0013, 0.0099]
    kBP = [1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043]
    kRP = [0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006]
    
    GBP_GRP = 1.2
    
    #E_B_V = 0.105 #M50
    E_B_V = 0.22 #0.22 #NGC3766
    
    A = 3.1 * E_B_V
    Ex_G = kG[0] + kG[1]*(GBP_GRP) + (kG[2]*(GBP_GRP)**2) + kG[3]*(GBP_GRP)**3 + (kG[4]*A) + (kG[5]*(A**2)) + (kG[6]*(GBP_GRP)*A)
    Ex_BP = kBP[0] + kBP[1]*(GBP_GRP) + (kBP[2]*(GBP_GRP)**2) + (kBP[3]*(GBP_GRP)**3) + (kBP[4]*A) + (kBP[5]*(A**2)) + (kBP[6]*(GBP_GRP)*A)
    Ex_RP = kRP[0] + kRP[1]*(GBP_GRP) + (kRP[2]*(GBP_GRP)**2) + (kRP[3]*(GBP_GRP)**3) + (kRP[4]*A) + (kRP[5]*(A**2)) + (kRP[6]*(GBP_GRP)*A)
    
    print('Reddening in G = ', Ex_G)
    print('Reddening in BP = ', Ex_BP)
    print('Reddening in RP = ', Ex_RP)
    print('Reddening in BP-RP = ', Ex_BP - Ex_RP)
    return Ex_G, Ex_BP-Ex_RP

def PlotHRDReddening(hdu, dG, dBpRp): # dG/dBpRp are reddening
    validID = hdu[1].data['cluster1']
    bp_rpMag = hdu[1].data['phot_bp_mean_mag'][validID] - hdu[1].data['phot_rp_mean_mag'][validID] + dBpRp
    gMag = hdu[1].data['phot_g_mean_mag'][validID] + dG
    
    x = 0.5 # Start x pos of arrow
    y = 16  # start y pos of arrow
    
    plt.scatter(bp_rpMag, gMag, s=0.1) # Data
    plt.annotate("", xy=(x+dBpRp, y+dG), xytext=(x,y), arrowprops=dict(arrowstyle="->")) # arrow
    plt.xlabel('Bp-Rp Mag')
    plt.ylabel('G Mag')
    plt.ylim(np.nanmax(gMag), np.nanmin(gMag))
    plt.savefig('HRD_Arrow.png', dpi=300)
    plt.show
    
    
plt.ion()
hdu = fits.open('WFI_GAIA_BJ_NGC3766_MATCH.fits')
dG, dBpRp = Reddening()
#PlotHRDReddening(hdu, dG, dBpRp)