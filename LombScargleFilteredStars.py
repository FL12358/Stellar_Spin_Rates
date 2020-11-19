import numpy as np
import matplotlib.pylab as plt
import math
#from astropy.timeseries import LombScargle
from astropy.stats import LombScargle
from astropy.io import fits
from matplotlib.ticker import FormatStrFormatter



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


def AppToAbs(appMag, hdu, freqs):
    dist = hdu[1].data['rest'][np.nonzero(freqs)]
    return appMag - 5*np.log10(dist) + 5


def ValidPowerVal(minPower, freq, power):
    d = 0.2
    a1 = 1/1.1
    a2 = 1/0.5
    
    if(1/freq < 0.2 or 1/freq > 40): # Valid period range
        return 0
    elif(power < minPower): # below low limit always invalid
        return 0
    elif(freq < a1+d and freq > a1-d): # inside 1 day alias - invalid
        return 0
    elif(freq < a2+d/2 and freq > a2-d/2): # inside 0.5 day alias - invalid
        return 0
    else:
        return 1
    
def FilterMedMag(freqs, hdu):
    index = np.nonzero(freqs)
    medMag = hdu[1].data['medflux'][index]
    print(np.shape(medMag))
    for i in range(len(medMag)):
        if(medMag[i] <= 14):
            freqs[i] = 0
    
    return freqs


def LSWrapper(time, flux, minPower): # Returns frequency/power of input time/flux arrays
    freq,power = LombScargle(time,flux).autopower(minimum_frequency=0.02, 
                                                maximum_frequency=5,
                                                samples_per_peak = 15)
    
    maxFreq = freq[power.argmax()]
    peakPower = power[power.argmax()]
    if ValidPowerVal(minPower, maxFreq, peakPower): # peakPower > minPower:
        return maxFreq,peakPower
    else:
        return 0,0
    
    
def LSData(hdu, minPower): # Returns array of valid frequencies or 0 if star not valid (length of number of stars)
    freqs = np.zeros(len(hdu[1].data['hjd']))
    power = np.zeros(len(hdu[1].data['hjd']))
    i = 0
    while i < len(hdu[1].data['hjd']):
        if i%100 == 0: print(i)
        #if(hdu[1].data['GAIA_Filter'][i] and not hdu[1].data['Cluster1'][i]):# if member of bool array
        #if(hdu[1].data['Cluster1'][i] and hdu[1].data['GAIA_Filter'][i]):
        #if(hdu[1].data['GAIA_Filter'][i]):
        freqs[i],power[i] = LSWrapper(hdu[1].data['hjd'][i],hdu[1].data['flux'][i], minPower)
        i += 1
    print("LS Length: " + str(np.count_nonzero(freqs)) + ' / ' + str(len(freqs)))
    return freqs,power

def PeriodFold(time,period):
    foldT = time/period - np.floor(time/period)
    return foldT

    
def DoubleAppendArrays(array, cutoff):
    right = array[cutoff:]
    left = array[:cutoff]
    array = np.append(right,array)
    array = np.append(array,left)
    return array


def FindCutOff(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
    
    
def LSModel(time, flux, i):
    modelT = np.linspace(0,1)
    modelF = LombScargle(time, flux).model(modelT, 1)
    return modelT, modelF
    
    
def BetterPlotPhaseFold(freqs, hdu, i, ax):
    time = hdu[1].data['hjd'][i]
    flux = hdu[1].data['flux'][i]
    freq = freqs[i]
    
    period = 1/freq # Period of light curve
    foldT = PeriodFold(time,period)
    fluxErr = hdu[1].data['fluxerr'][i]
    
    
    flux = flux[np.argsort(foldT)]
    fluxErr = fluxErr[np.argsort(foldT)]
    foldT = foldT[np.argsort(foldT)]
    
    cutoff = FindCutOff(foldT, 0.5)

    rightT = foldT[cutoff:]
    leftT = foldT[:cutoff]
    foldT = np.append(rightT-1.0,foldT)
    foldT = np.append(foldT,leftT+1.0)
    
    flux = DoubleAppendArrays(flux, cutoff)
    fluxErr = DoubleAppendArrays(fluxErr, cutoff)
    
    return ax.errorbar(foldT, flux, yerr = fluxErr,
                ecolor='gray',linestyle='',
                marker='.', color='0', ms=1)

def PlotLightCurve(freqs, hdu, i, ax):
    time = hdu[1].data['hjd'][i]
    time = time - time[0]
    flux = hdu[1].data['flux'][i]
    return ax.scatter(time, flux, s=1, c='0')

def PlotPhaseFoldedLightCurves(freqs, hdu):
    i = 0
    while i < len(freqs):
        if(freqs[i]):
            BetterPlotPhaseFold(freqs, hdu, i)
        i += 1

def PhaseFoldMultiPlot(freqs, hdu, pageNo=0, rows=8, cols=4):
    count = pageNo*rows*cols
    
    fig, axs = plt.subplots(rows,cols, figsize = [8,11])#, sharex = True)
    fig.subplots_adjust(hspace = 0.3, wspace=0.3)
    xlims = axs[0,0].set_xlim(-0.5,1.5)
    
    index = np.argwhere(freqs)
    
    i,j = 0,0
    
    while i < rows:
        while j < cols:
            if count >= len(index): # if no more plots
                axs[i,j].axis('off')
            else:
                #PlotLightCurve(freqs, hdu, index[count,0], axs[i,j])
                BetterPlotPhaseFold(freqs, hdu, index[count,0], ax = axs[i,j])
                title = '[' + str(count+1) + '] Period: {:.2f} days'.format(1/freqs[index[count,0]])
                axs[i,j].set_title(title, fontsize = 7.0) # title size
                axs[i,j].axvspan(xlims[0], 0, alpha=0.2, color='0') # grey areas
                axs[i,j].axvspan(1, xlims[1], alpha=0.2, color='0') 
                axs[i,j].tick_params(axis = 'both', labelsize = 6.0) # small mag labels
                axs[i,j].set_xticklabels([])
                axs[i,j].set_xlim(-0.5,1.5)
                ylims = axs[i,j].set_ylim() # get y limits
                axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                axs[i,j].vlines([0,1],ylims[0],ylims[1], lw=0.75) # plot vertial lines
                axs[i,j].invert_yaxis()
                
                if j == 0:
                    axs[i,j].set_ylabel('Magnitude [mag]', fontsize = 7.0)
                
                if i == (rows-1) or count+cols >= len(index): # or last plot of each col
                    axs[i,j].set_xlabel('Phase', fontsize = 7.0)
                    axs[i,j].set_xticklabels(['-0.5','0.0','0.5','1','1.5'])
                
            j += 1
            count += 1
        i += 1
        j = 0
        
    
    filename = 'PhaseFoldMulti3766_' + str(pageNo) + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()

def PhaseFoldMultiWrapper(freqs, hdu):
    rows = 8
    cols = 4
    totalPlots = len(np.argwhere(freqs))
    print('Total Number of plots: ' + str(totalPlots))
    pageNo = 0
    while pageNo*rows*cols < totalPlots:
        PhaseFoldMultiPlot(freqs, hdu, pageNo = pageNo, rows = rows, cols = cols)
        pageNo += 1
    

def PhaseFoldConstant(hdu):
    minPower = 0.1
    index = np.genfromtxt('WFI_Pointers', dtype = int)
    flux = hdu[1].data['flux']
    time = hdu[1].data['hjd']
    freqs = np.zeros(np.shape(hdu[1].data['bp_rp'])[0])

    for i in index:
        freqs[i], power = LSWrapper(time[i], flux[i], minPower)

    PhaseFoldMultiWrapper(freqs, hdu)


def PeriodColourPlot(freqs, hdu):
    periods = 1/freqs[np.nonzero(freqs)]
    colour = hdu[1].data['bp_rp'][np.nonzero(freqs)]
    
    plt.figure()
    plt.scatter(colour,periods, s=0.5, c='0')
    plt.ylabel('Period (Days)')
    plt.xlabel('Colour (Bp-Rp)')
    plt.title('Period/Color plot ')
    plt.yscale('log')
    plt.savefig('PeriodColourPlot.png', dpi=200)
    plt.close


def PeriodDistancePlot(freqs, hdu):
    periods = 1/freqs[np.nonzero(freqs)]
    distance = hdu[1].data['rest'][np.nonzero(freqs)]
    distanceError = 0 #1/hdu[1].data['parallax_error'][np.nonzero(freqs)]
    
    plt.figure()
    plt.errorbar(distance,periods,
                 xerr = distanceError,
                 ecolor='gray', linestyle='',
                 marker='.', color='0', ms=3)
    
    #plt.vlines(1745,0,np.max(periods), color = 'red')
    plt.ylabel('Period [Days]')
    plt.xlabel('Distance [pc]')
    plt.yscale('log')
    plt.xlim(0,3000)
    plt.yticks([1,10], ['1','10'])
    plt.savefig('PeriodDistancePlot', dpi=200)
    plt.close
    
    
def ColourDistancePlot(freqs,hdu):
    distance = hdu[1].data['rest'][np.nonzero(freqs)]
    colour = hdu[1].data['bp_rp'][np.nonzero(freqs)]
    
    plt.figure()
    plt.scatter(distance, colour, s = 0.5, c='0')
    plt.xlabel('Distance')
    plt.ylabel('Colour')
    plt.savefig('ColourDistancePlot', dpi=200)
    plt.close
    
    
def MaxPowerPeakVsFreq(hdu):
    freq = np.empty(0)
    power = np.empty(0)
    
    time = hdu[1].data['hjd']
    flux = hdu[1].data['flux']
    
    for i in range(len(hdu[1].data['pointer'])): 
        if(i%10000==0): print(i)
        f,p = LombScargle(time[i],flux[i]).autopower(minimum_frequency=0.02, 
                                                     maximum_frequency=5,
                                                     samples_per_peak = 15)
        freq = np.append(freq,f[p.argmax()])
        power = np.append(power,p[p.argmax()])
            
    plt.figure()
    plt.scatter(1/freq,power, s=0.1, c='0', alpha=0.1)
    plt.ylabel('LS Peak Power')
    plt.xlabel('Period')
    #plt.xlim(0,70)
    plt.xscale('log')
    plt.show
    plt.savefig('MaxPowerPeakVsPeriodLog.png', dpi=500)
    plt.close
    return freq,power


def PlotNoiseOnFoldedCurve(freqs, hdu): # Plot contstant star with folding from period detected
    print('PlotNoiseOnFoldedCurve')
    n = 70933
    const = 1000
    time = hdu[1].data['hjd'][n]    #heliocentric julian day
    time = time-time[0] #in days
    flux = hdu[1].data['flux'][const]
    foldP = 1/freqs[n] # Period of star
    foldT = PeriodFold(time,foldP)
    # Maybe add maximum time space filter here?
    
    title = 'Period: {:.2f} days (Constant star folded)'.format(foldP)
    
    plt.figure()
    plt.errorbar(foldT, flux,
                yerr=hdu[1].data['Fluxerr'][n],
                ecolor='gray',linestyle='',
                marker='.', color='0', ms=3)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.xlim(0,1)
    plt.xlabel('Phase')
    plt.ylabel('Magnitude')
    plt.show
    plt.close
    print('end')


def MedMagRmsPeriodic(freqs, hdu):
    medMag = hdu[1].data['medflux'][hdu[1].data['GAIA_Filter']]
    rms = hdu[1].data['rms'][hdu[1].data['GAIA_Filter']]
    freqs[freqs != 0] = 1
    color = freqs[hdu[1].data['GAIA_Filter']]

    rms,medMag,color = RemoveNaN(rms,medMag,color)
    rms,medMag,color = RemoveZeroes(rms,medMag,color)
    
    size = color
    size[size == 1] = 1
    size[size == 0] = 0
    
    plt.figure()
    plt.yscale('log')
    plt.scatter(medMag,rms,c = '0', s=1, alpha = 0.1)
    plt.scatter(medMag,rms,c = 'red', s=size, alpha = 1)
    plt.title('MedMag Vs Mag rms (periodic stars highlighted)')
    plt.xlabel('Median Magnitude')
    plt.ylabel('RMS')
    plt.xlim(11,18)
    
    plt.savefig('MedMagVsRmsPeriodic.png', dpi=300)
    plt.show
    plt.close
    
    
def ColorGMagPeriodPlot(freqs, hdu): # 23 Bp, 24 Rp, 22 G
    iso = np.genfromtxt('MIST_iso_1Gyr_Av0.iso.cmd')
    isoColor = iso[:,23] - iso[:,24] + 0.44
    isoGMag = iso[:,22] + 0.765 #+ 11.14
    
    periods = 1/freqs[np.nonzero(freqs)]
    bpRp = hdu[1].data['phot_bp_mean_mag'][np.nonzero(freqs)] - hdu[1].data['phot_rp_mean_mag'][np.nonzero(freqs)]
    Gmag = hdu[1].data['phot_g_mean_mag'][np.nonzero(freqs)]
    Gmag = AppToAbs(Gmag, hdu, freqs)
    
    plt.figure()
    plt.xlim(0.5,3.0)
    plt.ylim(2.5,11)
    plt.gca().invert_yaxis()
    plt.ylabel(r'Absolute Magnitude ($G$)')
    plt.xlabel(r'Color ($G_{BP}-G_{RP}$)')

    plt.plot(isoColor, isoGMag, color='0', alpha  = 0.5)
    plt.scatter(bpRp, Gmag, cmap = 'jet', c = periods, s = 2.5)
    plt.colorbar(label = 'Period (days)')
    
    plt.savefig("HRPeriodPlotFieldAbs.png", dpi = 300)
    plt.show
    plt.close
    
    
def PeriodHist(freqs, hdu):
    periods = 1/freqs[np.nonzero(freqs)]
    
    plt.figure()
    plt.ylabel('Percentage Occurance')
    plt.xlabel('Period (Days)')
    plt.hist(periods, bins = 18, color = '0', density = False)
    plt.savefig('PeriodOfRecoveredHIST.png', dpi=200)
    plt.close


def MagnitudeHist(freqs, hdu):
    mags = hdu[1].data['medflux'][np.nonzero(freqs)]
    
    plt.figure()
    plt.ylabel('Percentage Occurance')
    plt.xlabel('Magnitude')
    plt.hist(mags, bins = 15, color = '0', density = False)
    plt.savefig('MagnitudeOfRecoveredHIST.png', dpi=200)
    plt.close
    
    plt.figure()
    
    
def MQPeriodColor(freqs, hdu, mQData):
    periodsF = 1/freqs[np.nonzero(freqs)]
    colourF = hdu[1].data['bp_rp'][np.nonzero(freqs)]
    
    periodsMQ = mQData[1].data['Prot_MQ14']
    colourMQ = mQData[1].data['bp_rp_g']
    
    plt.figure()
    
    color = hdu[1].data['Cluster1'][np.nonzero(freqs)]
    
    plt.scatter(colourMQ,periodsMQ, s=0.5, alpha = 0.1, c = '0')
    plt.scatter(colourF,periodsF, s=2, c = color, cmap = 'bwr')
    
    plt.ylabel('Period (Days)')
    plt.xlabel('Colour (Bp-Rp)')
    plt.title('NGC3766 with MQ stars')
    plt.yscale('log')
    plt.ylim(0.1,100)
    plt.xlim(0.5,2.5)
    plt.savefig('mcQ_PeriodColourPlot', dpi=200)
    plt.close
    
    
def SaveDataToFile(freqs, hdu, filename): # Saves index, periods, colour and distance to text file
    index = np.nonzero(freqs)[0]
    periods = 1/freqs[index]
    color = hdu[1].data['bp_rp'][index]
    distance = hdu[1].data['rest'][index]
    absMag = AppToAbs(hdu[1].data['phot_g_mean_mag'][index], hdu, freqs)
    print(index)
    data = np.stack((periods, color, distance, absMag, index), axis=1)
    np.savetxt(filename, data, delimiter=',')
    
    
def FileToData(filename):
    data = np.transpose(np.genfromtxt(filename, delimiter=','))
    return data[0],data[1],data[2],data[3],data[4].astype(int)

def StdDevAmpPeriodic(hdu, index):
    stdDevs = np.std(hdu[1].data['flux'][np.nonzero(freqs)], axis=1)
    print(len(stdDevs))
    
    bins=np.logspace(-3.0,1.0, num = 25)
    print(bins)
    plt.xscale('log')
    plt.hist(stdDevs, bins = bins)
    plt.ylabel('Occurance')
    plt.xlabel('Std Dev of Amp')
    plt.savefig('AmpStdPeriodicStars.png', dpi=200)
    
    
def NGC869DataStuff():
    hdu = fits.open('Ground_Gaia_Matched_Field2.fits')
    minPower = 0.478
    freqs, power = LSData(hdu, minPower)
    
    SaveDataToFile(freqs, hdu, 'NGC869_Data.txt')
    return FileToData('NGC869_Data.txt')
    
    
    
#plt.ioff()
plt.ion()
#mQData = fits.open('MQ14_withGaia.fits')
hdu = fits.open('WFI_GAIA_BJ_NGC3766_MATCH.fits')
#hdu = fits.open('cluster_matched.fits')
#hdu = fits.open('Ground_Gaia_Matched_Field2.fits')
#hdu = fits.open('field_FINAL_2323.fits')

#MaxPowerPeakVsFreq(hdu)
'''
minPower = 0.429
#minPower = 0.478 # 869
minPower = 0.389
# NGC3766 distance = 1745pc
freqs,powers = LSData(hdu, minPower)
freqs = FilterMedMag(freqs, hdu)
np.savetxt('freqArray2323.txt', freqs)
'''



freqs = np.genfromtxt('freqArray3766.txt')

#BetterPlotPhaseFold(freqs, hdu, i)

#data = np.nonzero(freqs)
#MedMagRmsPeriodic(freqs, hdu)
#PlotNoiseOnFoldedCurve(freqs,hdu)
#PeriodColourPlot(freqs, hdu)
#PeriodDistancePlot(freqs, hdu)
#ColourDistancePlot(freqs,hdu)
#MaxPowerPeakVsFreq(hdu)
#PlotPhaseFoldedLightCurves(freqs, hdu)

#PhaseFoldMultiWrapper(freqs, hdu)
#PhaseFoldConstant(hdu)
#ColorGMagPeriodPlot(freqs, hdu)
#MagnitudeHist(freqs, hdu)
#PeriodHist(freqs, hdu)
periods,color,distance,absMag,index = NGC869DataStuff()


#MQPeriodColor(freqs, hdu, mQData)


#std = StdDevAmpPeriodic(hdu, freqs)

#filename = 'Period_colour_distance_absMag_NGC3766_Field.txt'
#SaveDataToFile(freqs, hdu, filename)
#periods,color,distance,absMag = FileToData(filename)



hdu.close