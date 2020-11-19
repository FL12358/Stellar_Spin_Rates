import numpy as np
import matplotlib.pylab as plt
from astropy.stats import LombScargle
from astropy.io import fits
from math import isclose
from notificator import Notificate
import time

MAXPERIOD = 50

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

def DAVENPORTFilters(hdu):
    index = np.arange(len(hdu[1].data['medFlux']))
    
    validId = np.logical_and((1/hdu[1].data['parallax_over_error'] < 0.1), (1/hdu[1].data['phot_g_mean_flux_over_error'] < 0.01))
    validId = np.logical_and(validId, (1/hdu[1].data['phot_bp_mean_flux_over_error'] < 0.01))
    validId = np.logical_and(validId, (1/hdu[1].data['phot_rp_mean_flux_over_error'] < 0.01))
    
    return index[validId]
    

def MedMagVsRmsFilter(hdu, index):
    cutOffPerc = 10
    medMag = np.empty(0)
    rms = np.empty(0)
    pointer = np.empty(0)
    fileIndex = np.arange(len(hdu[1].data['rms']))
    bins = 100
    
    for i in range(len(hdu[1].data['medflux'])):
        if(np.any(index == i) and (hdu[1].data['medflux'][i]) >= 14): # select valid stars
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
    validArraySplit = np.array_split(np.zeros(len(rms)),bins)
    cutOff = np.zeros(bins)    # Array of cutoff values for the bins

    for i in range(len(cutOff)):    # Calculate cutoff value for each bin
        cutOff[i] = np.nanpercentile(rmsSplit[i], cutOffPerc)
    
    #Loop through cutoff and rmsSplit[i] --> inner loop rmsSplit[i][j]
    for i in range(len(cutOff)): # Loops over each bin
        for j in range(len(rmsSplit[i])): # Loops through each star
            if(rmsSplit[i][j] < cutOff[i]):
                validArraySplit[i][j] = 1
        validArray = np.append(validArray, validArraySplit[i])
        
    rmsValid = np.extract(validArray, rms)    # Arrays of filtered stars
    medMagValid = np.extract(validArray, medMag)
    pointerValid = np.extract(validArray, pointer).astype(int)
    rmsValid = rmsValid[np.argsort(pointerValid)]    # sorts the arrays into ascending 'pointer'
    medMagValid = medMagValid[np.argsort(pointerValid)]  # Un-shuffles arrays
    pointerValid = pointerValid[np.argsort(pointerValid)]
    return pointerValid

    
def LSWrapper(time, flux):
    freq, power = LombScargle(time,flux).autopower(samples_per_peak = 5)
    return freq, power


def CreateSyntheticLC(index, hdu, period):
    time = hdu[1].data['hjd'][index] # Array
    time = time - time[0]
    noise = hdu[1].data['flux'][index]

    ampMax = 0.1 # max mag change for signal
    ampMin = 0.001
    sinAmp = np.random.uniform(ampMin,ampMax)
    
    flux = noise + sinAmp*np.sin(2*np.pi*time/period) # Noise + sin wave
    return time, flux


def InputVsOutputPeriods(hdu, index):
    samplesPerStar = 1000                        ### SAMPLES PER STAR
    print('Total Light Curves:', samplesPerStar*len(index))
    pIn = MAXPERIOD*(np.random.rand(samplesPerStar*len(index)))**10
    
    pOut = np.empty(0)
    powers = np.empty(0)
    
    a = 0
    for i in index:
        for j in range(samplesPerStar):
            if(a%1000 == 0): print(a)
            time, flux = CreateSyntheticLC(i, hdu, pIn[a])
            a += 1
            freq, power = LSWrapper(time,flux)
            pOut = np.append(pOut, 1/freq[power.argmax()])
            powers = np.append(powers, max(power))
    
    validIndex = np.zeros(len(pIn), dtype=int)  #valid index stuff colour the points red or green based on how close to the gradient=1 line they are
    for i in range(0,len(pIn)):
        if isclose(pIn[i], pOut[i], rel_tol=0.2):
           validIndex[i] = 1    # c=validIndex, cmap='RdYlGn'
    print("% close: ",100*np.count_nonzero(validIndex)/len(pIn))
    
    return pIn, pOut, powers


def CheckForMeasured(pIn, pOut, powers):
    th = 0.05
    minPower = 0.43
    
    Measured = np.where(powers > minPower)[0]
    NotMeasured = np.where(powers <= minPower)[0]
    Correct = np.where(abs(pIn-pOut)<=th*pIn)[0]
    Incorrect = np.where(abs(pIn-pOut)>th*pIn)[0]
    
    return Measured, NotMeasured, Correct, Incorrect


def PeriodBinSetup():
    binNo = 15 + 1
    binMin = 0.1
    binMax = MAXPERIOD
    
    bins = np.linspace(binMin, binMax, num = binNo)
    logbins = np.logspace(np.log10(binMin), np.log10(binMax), binNo)
    
    return bins, logbins

    
def BinWidthsCalc(bins, logbins, pIn, pOut):
    inputTotal,base = np.histogram(pIn, bins=bins) 
    outputTotal,base = np.histogram(pOut, bins=bins) 
    
    inputTotal_l,base_l = np.histogram(pIn, bins=logbins) 
    outputTotal_l,base_l = np.histogram(pOut, bins=logbins)
    
    bt=(base[:-1]+base[1:])/2. 
    wB=np.zeros([len(bt)])
    
    for i in range(len(bt)-1): 
        wB[i] = bt[i+1] - bt[i]
    wB[-1] = wB[-2]
    
    bt_l = (base_l[:-1] + base_l[1:])/2. 
    wB_l = np.zeros([len(bt_l)])
    
    for i in range(len(bt_l)-1): 
        wB_l[i] = 1.8*bt_l[i+1]
    wB_l[-1] = 1.5*wB_l[-2] # adjust here for end bin width
    
    return bt, wB, bt_l, wB_l


def PeriodCompletenessPlot(pIn,pOut, powers, ax):
    Measured, NotMeasured, Correct, Incorrect = CheckForMeasured(pIn, pOut, powers)
    bins, logbins = PeriodBinSetup()
    bt, wB, bt_l, wB_l = BinWidthsCalc(bins, logbins, pIn, pOut)
    
    inputTotal,base = np.histogram(pIn, bins=bins)
    Comp,base = np.histogram(pIn[list(set(Measured) & set(Correct))], bins = bins)

    return ax.bar(bt,100*Comp/inputTotal,width=wB,edgecolor='k',linewidth=.5,color='orange',alpha=0.5)

    
def PeriodCompletenessPlotLog(pIn,pOut, powers, ax):
    Measured, NotMeasured, Correct, Incorrect = CheckForMeasured(pIn, pOut, powers)
    bins, logbins = PeriodBinSetup()
    bt, wB, bt_l, wB_l = BinWidthsCalc(bins, logbins, pIn, pOut)
    
    inputTotal_l,base_l = np.histogram(pIn, bins=logbins) 
    Comp_l,base_l = np.histogram(pIn[list(set(Measured) & set(Correct))], bins = logbins)
    
    return ax.bar(bt_l,100*Comp_l/inputTotal_l,width = 0.15*wB_l,edgecolor='k',linewidth=.5,color='orange',alpha=0.5)


def PeriodReliabiltyContaiminationPlot(pIn, pOut, powers, ax):
    Measured, NotMeasured, Correct, Incorrect = CheckForMeasured(pIn, pOut, powers)
    bins, logbins = PeriodBinSetup()
    bt, wB, bt_l, wB_l = BinWidthsCalc(bins, logbins, pIn, pOut)
    
    totalMeasured,base=np.histogram(pOut[Measured], bins=bins)
    Reliable,base = np.histogram(pOut[list(set(Measured) & set(Correct))], bins=bins)
    Contaminant,base = np.histogram(pOut[list(set(Measured) & set(Incorrect))], bins=bins)

    return ax.bar(bt,100*Reliable/totalMeasured, width=wB, edgecolor='k', linewidth=.5, color='orange', alpha=0.5)

    
def PeriodReliabiltyContaiminationPlotLog(pIn, pOut, powers, ax):
    Measured, NotMeasured, Correct, Incorrect = CheckForMeasured(pIn, pOut, powers)
    bins, logbins = PeriodBinSetup()
    bt, wB, bt_l, wB_l = BinWidthsCalc(bins, logbins, pIn, pOut)
    
    totalMeasured_l,base_l = np.histogram(pOut[Measured], bins=logbins)
    Reliable_l,base_l = np.histogram(pOut[list(set(Measured) & set(Correct))],bins=logbins)
    Contaminant_l,base_l = np.histogram(pOut[list(set(Measured) & set(Incorrect))],bins=logbins)
    
    return ax.bar(bt_l,100*Reliable_l/totalMeasured_l, width=0.15*wB_l, edgecolor='k', linewidth=.5, color='orange', alpha=0.5)

    
def CompRelPlotter():
    fig, axs = plt.subplots(4,3, figsize = [8,11], sharey = True, sharex = 'row')
    fig.subplots_adjust(wspace=0.15)
    
    
    axs[0,0].set_ylim(0,100) # set 0 to 100% y limits
    axs[0,0].set_xlim(0,50) # set x lims
    axs[2,0].set_xlim(0,50) # set x lims
    axs[1,0].set_xlim(0.1,50)
    axs[3,0].set_xlim(0.1,50) # set x lims
    
    axs[3,0].set_xlabel('Period [days]')
    axs[3,1].set_xlabel('Period [days]')
    axs[3,2].set_xlabel('Period [days]')
    axs[0,0].set_title('NGC 2323', fontsize = 10.0)
    axs[0,1].set_title('NGC 3766', fontsize = 10.0)
    axs[0,2].set_title('NGC 869', fontsize = 10.0)
    axs[0,0].set_ylabel('Completeness [%]')
    axs[1,0].set_ylabel('Completeness [%]')
    axs[2,0].set_ylabel('Reliability [%]')
    axs[3,0].set_ylabel('Reliability [%]')
    
    logVals = [0.1,0.5,1,2,5,10,20,50]
    axs[1,0].set_xscale("log") # Set log scales
    axs[1,0].set_xticks(logVals)
    axs[1,0].set_xticklabels(logVals)
    axs[3,0].set_xscale("log")
    axs[3,0].set_xticks(logVals)
    axs[3,0].set_xticklabels(logVals)
    
    axs[0,0].set_xticks([0,10,20,30,40,50])
    axs[0,0].set_xticklabels([0,10,20,30,40,50])
    axs[2,0].set_xticks([0,10,20,30,40,50])
    axs[2,0].set_xticklabels([0,10,20,30,40,50])
    
    
    #NGC 2323 / M50
    pIn = np.genfromtxt('Input_Periods_M50.txt')
    pOut = np.genfromtxt('Output_Periods_M50.txt')
    powers = np.genfromtxt('Powers_M50.txt')
    
    PeriodCompletenessPlot(pIn, pOut, powers, axs[0,0])
    PeriodCompletenessPlotLog(pIn, pOut, powers, axs[1,0])
    PeriodReliabiltyContaiminationPlot(pIn, pOut, powers, axs[2,0])
    PeriodReliabiltyContaiminationPlotLog(pIn, pOut, powers, axs[3,0])
    

    
    # NGC 3766
    pIn = np.genfromtxt('Input_Periods.txt')
    pOut = np.genfromtxt('Output_Periods.txt')
    powers = np.genfromtxt('Powers.txt')
    
    PeriodCompletenessPlot(pIn, pOut, powers, axs[0,1])
    PeriodCompletenessPlotLog(pIn, pOut, powers, axs[1,1])
    PeriodReliabiltyContaiminationPlot(pIn, pOut, powers, axs[2,1])
    PeriodReliabiltyContaiminationPlotLog(pIn, pOut, powers, axs[3,1])
    

    
    # NGC 869
    pIn = np.genfromtxt('Input_Periods_hPer.txt')
    pOut = np.genfromtxt('Output_Periods_hPer.txt')
    powers = np.genfromtxt('Powers_hPer.txt')
    
    PeriodCompletenessPlot(pIn, pOut, powers, axs[0,2])
    PeriodCompletenessPlotLog(pIn, pOut, powers, axs[1,2])
    PeriodReliabiltyContaiminationPlot(pIn, pOut, powers, axs[2,2])
    PeriodReliabiltyContaiminationPlotLog(pIn, pOut, powers, axs[3,2])
    
    plt.savefig('Comp_rel_clusters.png', dpi = 200, bbox_inches='tight')
    

def PlotPinPout(pIn,pOut):
    validIndex = np.zeros(len(pIn), dtype=int)  #valid index stuff colour the points red or green based on how close to the gradient=1 line they are
    for i in range(0,len(pIn)):
        if isclose(pIn[i], pOut[i], rel_tol=0.2):
           validIndex[i] = 1
    
    def alias(P,n=1,m=1,df=1):
        return 1./abs((m/P)+(n*df))

    plt.scatter(pIn, pOut, s=1,cmap='RdYlGn',c=validIndex, alpha=0.05)
    
    aliasP = np.sort(pIn)
    alias1 = alias(aliasP,n=1,m=1,df=1)
    alias2 = alias(aliasP,n=-1,m=1,df=1)
    
    plt.plot(aliasP, alias1, c='0', alpha = 0.25, lw=3)
    plt.plot(aliasP, alias2, c='0', alpha = 0.25, lw=3)
    
    plt.xlabel('Input Period [days]')
    plt.ylabel('Detected Period [days]')
    plt.xscale('log')
    plt.yscale('log')
    plt.axis([0.1,50,0.1,70])
    plt.xticks([0.1,1,10], ['0.1','1','10'])
    plt.yticks([0.1,1,10], ['0.1','1','10'])
    plt.savefig("PinVsout1", dpi = 500, bbox_inches = 'tight')
    plt.show()
    plt.close
  

#Notificate('Starting Simulation')
exeTime = time.time()

hdu = fits.open('WFI_GAIA_BJ_NGC3766_MATCH.fits')
#hdu = fits.open('hPer_GAIA_Match.fits')
#hdu = fits.open('M50_GAIA_Match.fits')

#index = DAVENPORTFilters(hdu)
#np.savetxt('Constant_star_Indices', MedMagVsRmsFilter(hdu, index))



constantId = np.genfromtxt('Constant_star_Indices', dtype = 'int32')

#pIn, pOut, powers = InputVsOutputPeriods(hdu, constantId)

#np.savetxt('Input_Periods_3766.txt', pIn)
#np.savetxt('Output_Periods_3766.txt', pOut)
#np.savetxt('Powers_3766.txt', powers)


pIn = np.genfromtxt('Input_Periods_3766.txt')
pOut = np.genfromtxt('Output_Periods_3766.txt')
#powers = np.genfromtxt('Powers_3766.txt')
PlotPinPout(pIn,pOut)

#PeriodCompletenessPlot(pIn, pOut, powers)
#PeriodReliabiltyContaiminationPlot(pIn, pOut, powers)
#CompRelPlotter()



exeTime = time.time() - exeTime
timeStr = time.strftime("%H:%M:%S", time.gmtime(exeTime))
Notificate('Finished Running \n(' + timeStr + ')')




