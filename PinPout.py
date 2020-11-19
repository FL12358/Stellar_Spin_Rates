import numpy as np
import matplotlib.pylab as plt
from astropy.stats import LombScargle
from astropy.io import fits
from math import isclose

PERIODRANGE= 70 #Max val of pIn
NUMBER = 10000  #No of points
SIGNAL = 1 #Sin wave amplitude
NOISE = 1


def SinUneven(t,P): #Creates Sin curve array from time array input
    return SIGNAL*np.sin(2*np.pi*t/P)

def GNoise(values): #Adds gaussian noise to an input array
    noise = np.random.normal(scale=NOISE, size=len(values)) #array of length values with gaussian noise, -ve and +ve
    for i in range(len(values)):
        values[i] = values[i] + noise[i]
    return values

def MakepIn(maxVal,number): 
    return maxVal*(np.random.rand(number))**10    # **10 creates evenly spaced points in log-space

def MakepOut(pIn,time):
    pOut = np.empty(len(pIn))
    for P in range(len(pIn)):
        if P%1000 == 0: print(P)
        gn = GNoise(SinUneven(time,pIn[P]))
        freq,power = LombScargle(time,gn).autopower(minimum_frequency=0.02, 
                                maximum_frequency=5,
                                samples_per_peak = 15)
        pOut[P] = 1/(freq[power.argmax()])
    return pOut

def CleanArray(a,b):
    b = b[a != 0]
    a = a[a != 0]

    a = a[b != 0]
    b = b[b != 0]

    b = b[~np.isnan(a)]
    a = a[~np.isnan(a)]
    
    a = a[~np.isnan(b)]
    b = b[~np.isnan(b)]
    return a,b

def alias(P,n=1,m=1,df=1):
    return 1./abs((m/P)+(n*df))

def PvsPgraph(time):   #prints a period in vs out of the lombscargle with sin waves of various frequencies  
    pIn = MakepIn(PERIODRANGE,NUMBER)
    pOut = MakepOut(pIn,time)
    aliasP = np.sort(pIn)
    alias1 = alias(aliasP,n=1,m=1,df=1)
    alias2 = alias(aliasP,n=-1,m=1,df=1)
    
    validIndex = np.zeros(len(pIn), dtype=int)  #valid index stuff colour the points red or green based on how close to the gradient=1 line they are
    for i in range(0,len(pIn)):
        if isclose(pIn[i], pOut[i], rel_tol=0.1):
           validIndex[i] = 1    # c=validIndex, cmap='RdYlGn'
    print("% close: ",100*np.count_nonzero(validIndex)/len(pIn))
    
    plt.scatter(pIn, pOut, s=0.5,cmap='RdYlGn',c=validIndex, alpha=0.1)
    
    plt.plot(aliasP, alias1, c='0', alpha=0.5, linestyle = '--')
    plt.plot(aliasP, alias2, c='0', alpha=0.5, linestyle = '--')
    
    plt.xlabel('Input Period [days]')
    plt.ylabel('Detected Period [days]')
    plt.xscale('log')
    plt.yscale('log')
    plt.axis([0.1,70,0.1,70])
    plt.xticks([0.1,1,10], ['0.1','1','10'])
    plt.yticks([0.1,1,10], ['0.1','1','10'])
    plt.savefig("PinVsout1", dpi = 500)
    plt.show()
    plt.close

def PeakPowerVsFreq(time):
    pIn = MakepIn(PERIODRANGE,NUMBER)
    pOut = MakepOut(pIn,time)
    f = np.empty(len(pIn))
    peakPower = np.empty(len(pIn))
    
    for P in range(len(pIn)):
       gn = SIGNAL*np.sin(2*np.pi*time/pOut[P])
       freq,power = LombScargle(time,gn).autopower(minimum_frequency=0.01, 
                                                   maximum_frequency=10, 
                                                   samples_per_peak=5, 
                                                   nyquist_factor=10, 
                                                   method='fast')
       f[P] = freq[power.argmax()]
       peakPower[P] = power[power.argmax()]
       
    plt.scatter(f,peakPower, s=1)
    plt.savefig("PeakPowerVsFreq", dpi=500)
    plt.xlabel('Frequency')
    plt.ylabel('Peak Power')
    plt.show
    
def PInHistogram(time):
    pIn = MakepIn(PERIODRANGE,NUMBER)
    plt.hist(pIn,bins=40,range=(1,70),color='r')
    #plt.xscale('log')
    plt.savefig('pInHistogram')
    plt.show
    plt.close
    
def POutHistogram(time):
    pIn = MakepIn(PERIODRANGE,NUMBER)
    pOut = MakepOut(pIn,time)
    plt.hist(pOut,bins=40,range=(1,70),color=('0'))    #=np.logspace(0.01,2,num=40)
    #plt.xscale('log')
    plt.savefig('pOutHistogram2')
    plt.show
    plt.close
    
def PlotReliablePeriods(pIn,pOut,bins):
    pOut = pOut[np.argsort(pIn)]
    pIn = pIn[np.argsort(pIn)]
    validArray = np.isclose(pIn,pOut,rtol=0.05) # Bool array of if the pIn and Pout values are close
    pBin = np.empty(bins)
    recBin = np.empty(bins)
    arraySplit = np.array_split(validArray,bins)
    
    for i in range(bins):
        pBin[i] = (i + 0.5)*(np.max(pIn)/bins)
        recBin[i] = 100*np.count_nonzero(arraySplit[i])/len(arraySplit[i])
        
    plt.bar(pBin, recBin, 
            width=np.max(pIn)/bins,
            fc = (0.4, 0.4, 0.4, 0.5))
    
def PlotContaminatedPeriods(pIn,pOut,bins):
    # contaminated -> output outisde 10% of input period
    pIn = pIn[np.argsort(pOut)]
    pOut = pOut[np.argsort(pOut)]
    validArray = np.invert(np.isclose(pIn,pOut,rtol=0.05)) # Bool array of if the pIn and Pout values are not close
    pBin = np.empty(bins)
    recBin = np.empty(bins)
    arraySplit = np.array_split(validArray,bins)
    
    for i in range(bins):
        pBin[i] = (i + 0.5)*(np.max(pIn)/bins)
        recBin[i] = 100*np.count_nonzero(arraySplit[i])/len(arraySplit[i])
        
    plt.bar(pBin, recBin, 
            width=np.max(pIn)/bins,
            fc = (0.8, 0.2, 0.2, 0.5))


def PlotCompleteness(pIn, pOut):
    pIn = pIn[np.argsort(pOut)]
    pOut = pOut[np.argsort(pOut)]
    
    validArray = np.isclose(pIn,pOut,rtol=0.05)
    
    plt.hist(pIn[validArray], bins = 50)
    
    
def ReliabilityContaminationPlot(time):
    bins = 25
    pIn = MakepIn(PERIODRANGE,NUMBER)
    pOut = MakepOut(pIn,time)
    PlotCompleteness(pIn,pOut)
    
    #PlotReliablePeriods(pIn,pOut,bins)
    #PlotContaminatedPeriods(pIn,pOut,bins)
    
    plt.ylabel('Percentage Recovered')
    plt.xlabel('Period')
    plt.title('Reliability/Contamination (NGC3766) SNR: %.2f' % (SIGNAL/NOISE))
    plt.show
    plt.savefig('RelConPlotNGC3766', dpi=500)
    
def SignalToNoise(hdu):
    pointer = np.genfromtxt('WFI_Pointers').astype(np.int32)
    signal = hdu[1].data['medflux']
    noise = hdu[1].data['rms']
    
    signal = signal[(pointer)]
    noise = noise[(pointer)]
    signal, noise = CleanArray(signal,noise)
    snr = signal/noise
    return np.mean(snr)
    
    
#hdu = fits.open('lc_NGC2516_1_ctio')
#hdu = fits.open('lc_hPer_maidanak.fits')
hdu = fits.open('lc_NGC3766_wfi.fits')  #Max points 402 (eg @18), total stars 32103
#print(hdu[1].columns)

n = 18    #Star No
time = hdu[1].data['hjd'][n]    #heliocentric julian day
time = time-time[0] #in days

#PeakPowerVsFreq(time)
PvsPgraph(time)
#PInHistogram(time)
#POutHistogram(time)
#ReliabilityContaminationPlot(time)
#SNR = SignalToNoise(hdu)
#print(SNR)
hdu.close