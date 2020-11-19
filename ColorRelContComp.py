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


def MedMagVsRmsFilter(hdu):
    cutOffPerc = 10
    medMag = np.empty(0)
    rms = np.empty(0)
    pointer = np.empty(0)
    fileIndex = np.arange(len(hdu[1].data['rms']))
    bins = 100
    
    for i in range(len(hdu[1].data['medflux'])):
        if(hdu[1].data['GAIA_Filter'][i] and (hdu[1].data['medflux'][i]) >= 14): # select valid stars
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

   # plt.yscale('log')
   # plt.scatter(medMag,rms,marker=".",s=2,c=validArray,cmap='RdYlGn',alpha=0.5)
    #plt.ylim(0.001,10)
  #  plt.savefig('MedMagVsRmsBin', dpi=500)
   # plt.close
    
    rmsValid = np.extract(validArray, rms)    # Arrays of filtered stars
    medMagValid = np.extract(validArray, medMag)
    pointerValid = np.extract(validArray, pointer).astype(int)
    rmsValid = rmsValid[np.argsort(pointerValid)]    # sorts the arrays into ascending 'pointer'
    medMagValid = medMagValid[np.argsort(pointerValid)]  # Un-shuffles arrays
    pointerValid = pointerValid[np.argsort(pointerValid)]
    
    print(len(pointerValid), 'constant stars')
    return pointerValid

    
def LSWrapper(time, flux):
    freq, power = LombScargle(time,flux).autopower(samples_per_peak = 5)
    
    #freq = 1/np.linspace(0.1,50,1000) # 1 / periods
    #power = LombScargle(time, flux).power(freq)
    return freq, power


def CreateSyntheticLC(index, hdu, period):
    time = hdu[1].data['hjd'][index] # Array
    noise = hdu[1].data['flux'][index]
    color = hdu[1].data['rest'][index] # Change to other vals if needed
    time = time - time[0]
    
    ampMax = 0.1 # max mag change for signal
    ampMin = 0.001
    sinAmp = np.random.uniform(ampMin,ampMax)
    
    flux = noise + sinAmp*np.sin(2*np.pi*time/period) # Noise + sin wave
    
    return time, flux, color


def InputVsOutputPeriods(hdu, index):
    samplesPerStar = 100
    print('Total Light Curves:', samplesPerStar*len(index))
    pIn = MAXPERIOD*(np.random.rand(samplesPerStar*len(index))) # **10
    
    pOut = np.empty(0)
    powers = np.empty(0)
    colors = np.empty(0)
    
    a = 0
    for i in index:
        for j in range(samplesPerStar):
            if(a%1000 == 0): print(a)
            time, flux, color = CreateSyntheticLC(i, hdu, pIn[a])
            a += 1
            freq, power = LSWrapper(time,flux)
            pOut = np.append(pOut, 1/freq[power.argmax()])
            powers = np.append(powers, max(power))
            colors = np.append(colors, color)
    
    validIndex = np.zeros(len(pIn), dtype=int)  #valid index stuff colour the points red or green based on how close to the gradient=1 line they are
    for i in range(0,len(pIn)):
        if isclose(pIn[i], pOut[i], rel_tol=0.2):
           validIndex[i] = 1    # c=validIndex, cmap='RdYlGn'
    print("% close: ",100*np.count_nonzero(validIndex)/len(pIn))
    
    return pIn, pOut, powers, colors


def CheckForMeasured(pIn, pOut, powers):
    th = 0.05
    minPower = 0.43
    
    Measured = np.where(powers > minPower)[0]
    NotMeasured = np.where(powers <= minPower)[0]
    Correct = np.where(abs(pIn-pOut)<=th*pIn)[0]
    Incorrect = np.where(abs(pIn-pOut)>th*pIn)[0]
    
    return Measured, NotMeasured, Correct, Incorrect


def ColorBinSetup():
    binNo = 15
    binMin = 0
    binMax = 2500
    
    bins = np.linspace(binMin, binMax, num = binNo)
    logbins = np.logspace(np.log10(binMin), np.log10(binMax), binNo)
    
    return bins, logbins


def ColorBinWidthsCalc(bins, logbins, pIn, pOut):
    inputTotal,base = np.histogram(pIn, bins=bins) 
    outputTotal,base = np.histogram(pOut, bins=bins) 

    bt=(base[:-1]+base[1:])/2. 
    wB=np.zeros([len(bt)])
    
    for i in range(len(bt)-1): 
        wB[i] = bt[i+1] - bt[i]
    wB[-1] = wB[-2]
    
    return bt, wB


def ColorCompletenessPlot(pIn, pOut, powers, colors):
    Measured, NotMeasured, Correct, Incorrect = CheckForMeasured(pIn, pOut, powers) # same for period
    bins, logbins = ColorBinSetup() # Color bins
    bt, wB = ColorBinWidthsCalc(bins, logbins, pIn, pOut)

    inputTotal,base = np.histogram(colors, bins=bins) 
    Comp,base = np.histogram(colors[list(set(Measured) & set(Correct))], bins = bins)
    
    fig, ax = plt.subplots(figsize = (8,5))
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.27, top=0.98)
    
    ax0 = plt.subplot2grid((1,2),(0,0),colspan=1,rowspan=1)
    ax0.bar(bt,100*Comp/inputTotal,width=wB,
            edgecolor='k',linewidth=.5,
            color='orange',alpha=0.5)
    
    ax0.set_xlabel('Distance [pc]')
    ax0.set_ylabel('Completeness [%]')
    ax0.set_ylim(0,100)
    plt.xlim(0,2500)
    

def ColorReliabiltyContaiminationPlot(pIn, pOut, powers, colors):
    Measured, NotMeasured, Correct, Incorrect = CheckForMeasured(pIn, pOut, powers)
    bins, logbins = ColorBinSetup()
    bt, wB = ColorBinWidthsCalc(bins, logbins, pIn, pOut)
    
    totalMeasured,base=np.histogram(colors[Measured], bins=bins)
    Reliable,base = np.histogram(colors[list(set(Measured) & set(Correct))], bins=bins)
    Contaminant,base = np.histogram(colors[list(set(Measured) & set(Incorrect))], bins=bins)
    
    ax0 = plt.subplot2grid((1,2),(0,1),colspan=1,rowspan=1)
    ax0.bar(bt,100*Reliable/totalMeasured, width=wB, edgecolor='k', linewidth=.5, color='orange', alpha=0.5)
    ax0.bar(bt,100*Contaminant/totalMeasured, width=wB, edgecolor='k', linewidth=.5, color='red', alpha=0.5, zorder=10)
    ax0.set_ylim(0,100)
    ax0.set_xlim(0,2500)
    ax0.set_xlabel('Distance [pc]')
    ax0.set_ylabel('Reliability/Contamination [%]')
    plt.savefig('Color_CompRelCont.png', dpi=200)




def PeriodBinSetup():
    binNo = 20 + 1
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
        wB_l[i] = 2.1*bt_l[i+1]
    wB_l[-1] = 1.3*wB_l[-2]
    
    return bt, wB, bt_l, wB_l


def PeriodCompletenessPlot(pIn,pOut,powers):
    Measured, NotMeasured, Correct, Incorrect = CheckForMeasured(pIn, pOut, powers)
    bins, logbins = PeriodBinSetup()
    bt, wB, bt_l, wB_l = BinWidthsCalc(bins, logbins, pIn, pOut)
    
    inputTotal,base = np.histogram(pIn, bins=bins) 
    inputTotal_l,base_l = np.histogram(pIn, bins=logbins) 

    
    Comp,base = np.histogram(pIn[list(set(Measured) & set(Correct))], bins = bins)
    Comp_l,base_l = np.histogram(pIn[list(set(Measured) & set(Correct))], bins = logbins)
    
    fig, ax = plt.subplots(figsize = (8,5))
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.27, top=0.98)
    
    ax0 = plt.subplot2grid((1,2),(0,0),colspan=1,rowspan=1)
    ax0.bar(bt,100*Comp/inputTotal,width=wB,edgecolor='k',linewidth=.5,color='orange',alpha=0.5)
    ax0.set_xlabel('Period (days)')
    ax0.set_ylabel('Completeness (%)')
    ax0.set_ylim(0,100)
    
    ax0 = plt.subplot2grid((1,2),(0,1),colspan=1,rowspan=1)
    ax0.bar(bt_l,100*Comp_l/inputTotal_l,width = 0.1*wB_l,edgecolor='k',linewidth=.5,color='orange',alpha=0.5)
    ax0.set_xlabel('Period (days)')
    ax0.set_ylabel('Completeness (%)')
    ax0.set_xscale("log")
    ax0.set_xticks([0.2,0.5,1,2,10,20,50])
    ax0.set_xticklabels([0.2,0.5,1,2,10,20,50])
    ax0.set_ylim(0,100)
    ax0.set_xlim(0,MAXPERIOD)
    
    plt.savefig("Comp_ConstStar.png", dpi=200)

def PeriodReliabiltyContaiminationPlot(pIn, pOut, powers):
    Measured, NotMeasured, Correct, Incorrect = CheckForMeasured(pIn, pOut, powers)
    bins, logbins = PeriodBinSetup()
    bt, wB, bt_l, wB_l = BinWidthsCalc(bins, logbins, pIn, pOut)
    
    totalMeasured,base=np.histogram(pOut[Measured], bins=bins)
    Reliable,base = np.histogram(pOut[list(set(Measured) & set(Correct))], bins=bins)
    Contaminant,base = np.histogram(pOut[list(set(Measured) & set(Incorrect))], bins=bins)

    totalMeasured_l,base_l = np.histogram(pOut[Measured], bins=logbins)
    Reliable_l,base_l = np.histogram(pOut[list(set(Measured) & set(Correct))],bins=logbins)
    Contaminant_l,base_l = np.histogram(pOut[list(set(Measured) & set(Incorrect))],bins=logbins)

    fig, ax = plt.subplots(figsize = (8,5))
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.27, top=0.98)
    
    ax0 = plt.subplot2grid((1,2),(0,0),colspan=1,rowspan=1)
    ax0.bar(bt,100*Reliable/totalMeasured, width=wB, edgecolor='k', linewidth=.5, color='orange', alpha=0.5)
    ax0.bar(bt,100*Contaminant/totalMeasured, width=wB, edgecolor='k', linewidth=.5, color='red', alpha=0.5, zorder=10)
    ax0.set_ylim(0,100)
    ax0.set_xlim(0,MAXPERIOD)
    ax0.set_xlabel('Period (days)')
    ax0.set_ylabel('Reliability/Contamination (%)')
    
    ax0 = plt.subplot2grid((1,2),(0,1),colspan=1,rowspan=1)
    ax0.bar(bt_l,100*Reliable_l/totalMeasured_l, width=0.1*wB_l, edgecolor='k', linewidth=.5, color='orange', alpha=0.5)
    ax0.bar(bt_l,100*Contaminant_l/totalMeasured_l, width=0.1*wB_l, edgecolor='k', linewidth=.5, color='red', alpha=0.5, zorder=10)
    ax0.set_ylim(0,100)
    ax0.set_xlabel('Period (days)')
    ax0.set_xscale("log")
    ax0.set_xticks([0.2,0.5,1,2,10,20,50])
    ax0.set_xticklabels([0.2,0.5,1,2,10,20,50])
    
    plt.savefig("RelCont_ConstStar.png", dpi=200)
    
def InputAmpHist(index):
    sinAmps = np.random.uniform(0.001,0.1,len(index))
    plt.hist(sinAmps, color = 'orange', bins = 50, edgecolor='black', linewidth=1)
    plt.ylabel('No of light curves')
    plt.xlabel('Amplitude')
    plt.savefig('LightCurveAmpHist.png')
    
    plt.close
    

def StdDevLightCurves(index, hdu, pIn):
    stdDevsAll = np.empty(0)
    stdDevsP = np.empty(0)
    samplesPerStar = 1
    minPower = 0.5
    #bins = np.logspace(0, 0.5, num = 50)
    bins = 50
    a = 0
    print('Total Light Curves:', samplesPerStar*len(index))
    
    for i in index:
        for j in range(samplesPerStar):
            time, flux, color = CreateSyntheticLC(i, hdu, pIn[a])
            if(a % 1000 == 0): print(a)
            stdDevsAll = np.append(stdDevsAll, np.std(flux))
            freq, power = LSWrapper(time, flux)
            if(max(power) > minPower):    
                stdDevsP = np.append(stdDevsP, np.std(flux))
            a += 1
            
    print('Total Light Curves:', len(stdDevsAll))
    print('Periodic Light Curves:', len(stdDevsP))        
            
    plt.hist(stdDevsAll, bins = bins)
    plt.hist(stdDevsP, bins = bins)
    plt.ylabel('Occurance')
    plt.xlabel('std Dev mags')
    plt.xlim(0,0.5)
    plt.savefig('StdDevHist.png', dpi=200)


def InputOutputPlot(pIn,pOut):
    plt.figure()
    
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(pIn,pOut)
    plt.ylim(0,50)


#Notificate('Starting Simulation')
exeTime = time.time()

hdu = fits.open('WFI_GAIA_BJ_NGC3766_MATCH.fits')

np.savetxt('WFI_Constant_star_Indices', MedMagVsRmsFilter(hdu))
constantId = np.genfromtxt('WFI_Constant_star_Indices', dtype = 'int32')

'''
pIn, pOut, powers, colors = InputVsOutputPeriods(hdu, constantId)
np.savetxt('Input_Periods.txt', pIn)
np.savetxt('Output_Periods.txt', pOut)
np.savetxt('Powers.txt', powers)
np.savetxt('Colors.txt', colors)

'''
pIn = np.genfromtxt('Input_Periods.txt')
pOut = np.genfromtxt('Output_Periods.txt')
powers = np.genfromtxt('Powers.txt')
colors = np.genfromtxt('Colors.txt')

InputOutputPlot(pIn, pOut)

#InputAmpHist(constantId)
#ColorCompletenessPlot(pIn, pOut, powers, colors)
#ColorReliabiltyContaiminationPlot(pIn, pOut, powers, colors)
#StdDevLightCurves(constantId, hdu, pIn)







exeTime = time.time() - exeTime
timeStr = time.strftime("%H:%M:%S", time.gmtime(exeTime))
Notificate('Finished Running \n(' + timeStr + ')')




