import numpy as np
from astropy.io import fits
from astropy.stats import LombScargle
import matplotlib.pylab as plt


def FileToData(filename):
    data = np.transpose(np.genfromtxt(filename, delimiter=','))
    return data[0],data[1],data[2],data[3],data[4].astype(int)


periods,color,distance,absMag,index = FileToData('NGC869_Data.txt')
# Getting data from the file
# 'index' is the row values of the periodic stars in the fits file
print(index)


# For example getting some light cuve data, names might be different for your fits file
hdu = fits.open('Ground_Gaia_Matched_Field2.fits')

fluxes = hdu[1].data['flux'][index]
times = hdu[1].data['hjd'][index]

starId = 5 # index of star (0 to 8)
freq, power = LombScargle(times[starId], fluxes[starId]).autopower()

print(periods[starId]) # Period of selected star

plt.plot(freq,power, lw=1) # Plot periodogram
plt.vlines(1/periods[starId],0,1,
           lw=5, alpha = 0.5, color='red') # Line on the frequency corresponding to the star


