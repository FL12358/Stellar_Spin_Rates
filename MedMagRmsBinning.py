import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
from astropy.stats import LombScargle

def RemoveNaN(a,b,c):   #Determines NaNs in f and creates new shorter arrays without NaNs
    validIndex = np.isnan(a)
    a1 = a[~validIndex]
    b1 = b[~validIndex]
    c1 = c[~validIndex]
    return a1,b1,c1

def RemoveZeroes(a,b,c):
    a1 = a[a != 0]
    b1 = b[a != 0]
    c1 = c[a != 0]
    return a1,b1,c1


def MedMagVsRmsFilter(hdu):
    cutOffPerc = 10
    medMag = np.empty(0)
    rms = np.empty(0)
    pointer = np.empty(0)
    fileIndex = np.arange(len(hdu[1].data['rms']))
    bins = 100
    
    for i in range(len(hdu[1].data['medflux'])):
        if(hdu[1].data['GAIA_Filter'][i] and (hdu[1].data['medflux'][i]) >= 13.5): # select valid stars
            medMag = np.append(medMag,hdu[1].data['medflux'][i])
            rms = np.append(rms,hdu[1].data['rms'][i])
            pointer = np.append(pointer, fileIndex[i])
            
    rms,medMag,pointer = RemoveNaN(rms,medMag,pointer)   # Clean data
    rms,medMag,pointer = RemoveZeroes(rms,medMag,pointer)
    rms = rms[np.argsort(medMag)]
    pointer = pointer[np.argsort(medMag)]
    medMag = medMag[np.argsort(medMag)]    # Sort arrays by increasing median mag
    validArray = np.zeros(0, dtype = np.int8)    # If above or below cutoff
    
    rmsSplit = np.array_split(rms,bins)    # List of arrays that contain rms values of each bin
    validArraySplit = np.array_split(np.zeros(len(rms)),bins) # array of zeroes same as rmsSplit
    cutOff = np.zeros(bins)    # Array of cutoff values for the bins

    for i in range(len(cutOff)):    # Calculate cutoff value for each bin
        cutOff[i] = np.nanpercentile(rmsSplit[i], cutOffPerc)
    
    #Loop through cutoff and rmsSplit[i] --> inner loop rmsSplit[i][j]
    for i in range(len(cutOff)): # Loops over each bin
        for j in range(len(rmsSplit[i])): # Loops through each star
            if(rmsSplit[i][j] < cutOff[i]):
                validArraySplit[i][j] = 1
        validArray = np.append(validArray, validArraySplit[i])
    
    pointerValid = np.extract(validArray, pointer).astype(int)
    pointerValid = pointerValid[np.argsort(pointerValid)]
    
    print(len(pointerValid), 'constant stars')
    return pointerValid
    
def ValidPowerVal(minPower, freq, power):
    d = 0.1
    a1 = 0.95
    a2 = 1/0.5
    c = 0.3
    if(np.isnan(power)):
        return 1
    if(1/freq < 0.2 or 1/freq > 40): # Valid period range
        return 0
    if(power > minPower+c): # above high limit is always valid
        return 1
    elif(power < minPower): # below low limit always invalid
        return 0
    elif(freq < a1+d and freq > a1-d): # inside 1 day alias - invalid
        return 0
    elif(freq < a2+d/2 and freq > a2-d/2): # inside 0.5 day alias - invalid
        return 0
    else:
        return 1


def MaxPowerPeakVsFreq(pointer,hdu):
    freq = np.empty(0)
    power = np.empty(0)

    for i in pointer:
        f,p = LombScargle(hdu[1].data['hjd'][i],hdu[1].data['flux'][i]).autopower(minimum_frequency=0.02, 
                                                maximum_frequency=5,
                                                samples_per_peak = 15)
        
        if(ValidPowerVal(0.0,f[p.argmax()],p[p.argmax()])):
            freq = np.append(freq,f[p.argmax()])
            power = np.append(power,p[p.argmax()])

    return freq,power

def CumFreqDistPowerPeak(power, percentile): # Cumfreq of lowest rms stars
    power = power[np.nonzero(power)]
    cutoff = np.nanpercentile(power, percentile)
    print(cutoff)
    print(len(power))
    
    plt.hist(power,
             bins = 500,
             cumulative = True,
             histtype = 'step',
             density = True)
    
    plt.hlines(percentile/100,0,cutoff, lw = 1)
    plt.vlines(cutoff,0,percentile/100, lw = 1)
    plt.xlim(0,0.5)
    plt.show()
    plt.savefig('CumFreqOfPowerPeakConstant.png', dpi=200)
    plt.close
    return cutoff

percentile = 97
hdu = fits.open('WFI_GAIA_BJ_NGC3766_MATCH.fits')
pointer = MedMagVsRmsFilter(hdu)
np.savetxt('WFI_Pointers',pointer)
f,power = MaxPowerPeakVsFreq(pointer,hdu)
data = np.transpose(np.stack((1/f,power)))
cutoff = CumFreqDistPowerPeak(power, percentile)




