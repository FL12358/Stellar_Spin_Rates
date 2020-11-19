import numpy as np
import matplotlib.pylab as plt
from matplotlib.cm import ScalarMappable
from astropy.stats import LombScargle
import math
from astropy.io import fits
import matplotlib.patches as mpatches



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

def FileToData(filename):
    data = np.transpose(np.genfromtxt(filename, delimiter=','))
    return data[0],data[1],data[2],data[3] # period,color,distance,absMag




def LSUneven():
    period = 2
    tMax = 10
    
    time = tMax*np.random.rand(50)
    noise = np.random.normal(0.0,0.2,len(time))
    noise = 0
    time = np.sort(time)
    flux = noise + 1*np.sin(2*np.pi*time/period)
    timeS = np.arange(0,tMax,0.01)
    fluxS = 1*np.sin(2*np.pi*timeS/period)
    freq, power = LombScargle(time,flux).autopower()
    
    
    fig, axs = plt.subplots(2,1, gridspec_kw={'hspace': 0.5}, sharex = False)
    
    axs[0].scatter(time,flux, s=3, c='0')
    axs[0].plot(timeS,fluxS,lw=1)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')
    
    
    axs[1].plot(freq,power, lw=1, c='0')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Power')
    axs[1].vlines(1/period,-0.1,1.5, colors = 'red', linestyles = 'dotted')
    axs[1].set_xlim(0,10)
    axs[1].set_ylim(0,1.05)
    axs[1].set_xticks([0,1/period,2,4,6,8,10])
    
    plt.savefig('LSP_example.png', dpi=300, bbox_inches='tight')
    
    
def TestSunSpot():
    time = np.arange(0,100,0.1)

    rStar = 1.0
    rSpot = 0.3
    period = 25
    
    
    aStar = math.pi * (rStar)**2
    aSpot = math.pi * (rSpot)**2 * np.cos(2*math.pi*time/period)
    
    fSin = 1 - aSpot/aStar
    fStar = 1 - aSpot/aStar # 1- spotsize
    fStar = np.where(fStar<=1, fStar, 1)
    
    fig, axs = plt.subplots(3,1, gridspec_kw={'hspace': 0.2}, sharex = True)
    
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time")
    plt.ylabel("Relative Magnitude")
    
    
    axs[0].set_ylim(0.9,1.1)
    axs[0].plot(time, fStar, lw=1, c = '0')
    axs[0].plot(time, fSin, lw=3, alpha = 0.5) # sin curve

    axs[0].set_ylim(0.9,1.1)
    axs[1].set_ylim(0.9,1.02)
    axs[2].set_ylim(0.9,1.02)
    
    fTransit = np.ones(len(time))
    fEclipse = np.full(len(time), 0.91)
    periodArr = np.full(len(time), period)
    
    fTransit = np.where(np.isclose(np.mod(time-period/10, periodArr), period, atol = 5),
                        fEclipse, fTransit)
    axs[1].plot(time, fTransit, lw=1, c = '0')
    
    
    fEclipse2 = np.full(len(time), 0.99)
    fTransit = np.where(np.isclose(np.mod(time-115, periodArr), period, atol = 5),
                        fEclipse2, fTransit)
    
    axs[2].plot(time, fTransit, lw=1, c = '0')
    plt.savefig("LightCurve_Example.png", dpi=300, bbox_inches='tight')
    
    
    
def ObsWindows():
    hPer = fits.open('lc_hPer_maidanak.fits')
    wfi = fits.open('lc_NGC3766_wfi.fits')
    m50 = fits.open('lc_M50_ctio.fits')
    
    a,b,c = 1,1,1 # star indices
    timeHPer = hPer[1].data['hjd'][a] - hPer[1].data['hjd'][a][0]
    timewfi = wfi[1].data['hjd'][b] - wfi[1].data['hjd'][b][0]
    timeM50 = m50[1].data['hjd'][c] - m50[1].data['hjd'][c][0]
    
    fig, axs = plt.subplots(3,1, sharex = False,
                            gridspec_kw={'hspace': 0.75})
    
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time [Days]")
    for i in axs:
        i.tick_params(left=False, labelleft=False)
    
    width = 0.1
    axs[0].bar(timeHPer,1, width = width, color='0')
    axs[1].bar(timewfi,1, width = width, color='0')
    axs[2].bar(timeM50,1, width = 10*width, color='0')
    axs[0].set_title('NGC 2323')
    axs[1].set_title('NGC 3766')
    axs[2].set_title('NGC 869')
    
    plt.savefig('Cluster_obs_window.png', dpi=200, bbox_inches='tight')
    
    
def PlanckLaw():
    def B(L,T):
        h = 6.62607015*10**(-34)
        c = 3.0*10**(8)
        e = 2.71828
        k = 1.380649*10**(-23)
        return 2*h*c**2/(L**5) * 1/(e**(h*c/(L*k*T))-1)
    
    waves = np.linspace(1e-7,3e-6,1000)
    
    rad6k = B(waves,6000)
    rad5k = B(waves,5000)
    rad4k = B(waves,4000)
    rad3k = B(waves,3000)
    
    scaleWave = 1e9
    waves = waves*scaleWave
    
    maxFlux = np.amax(rad6k)
    rad6k /= maxFlux
    rad5k /= maxFlux
    rad4k /= maxFlux
    rad3k /= maxFlux
    
    sixK = plt.plot(waves, rad6k, c = 'dodgerblue')
    fiveK = plt.plot(waves,rad5k, c = 'gold')
    fourK = plt.plot(waves,rad4k, c='orange')
    threeK = plt.plot(waves,rad3k, c = 'red')

    
    bp = 532e-9 * scaleWave
    rp = 797e-9 * scaleWave
    xmin,xmax = plt.ylim(0,1.01)
    plt.vlines([bp,rp],xmin,xmax,linestyles = 'dotted')
    
    plt.legend((sixK[0],fiveK[0],fourK[0],threeK[0]),
               ('6000K','5000K','4000K','3000K'))
    
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Relative Flux')
    plt.savefig('PLACK_LAW.png',dpi=300, bbox_inches='tight')
    
    
    
def MedMagVsRmsFilter():
    hdu = fits.open('WFI_GAIA_BJ_NGC3766_MATCH.fits')
    cutOffPerc = 10
    medMag = np.empty(0)
    rms = np.empty(0)
    pointer = np.empty(0)
    fileIndex = np.arange(len(hdu[1].data['rms']))
    bins = 50

    for i in range(len(hdu[1].data['medflux'])):
        if(hdu[1].data['GAIA_Filter'][i] and (hdu[1].data['medflux'][i]) >= 13.75): # select valid stars
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
    
    for i in range(len(cutOff)): # Loops over each bin
        for j in range(len(rmsSplit[i])): # Loops through each star
            if(rmsSplit[i][j] < cutOff[i]):
                validArraySplit[i][j] = 1
        validArray = np.append(validArray, validArraySplit[i])

    #PLOT
    rms2 = hdu[1].data['rms'] # non-selected stars
    medMag2 = hdu[1].data['medflux'][rms2 != 0]
    rms2 = rms2[rms2 != 0]
    
    plt.yscale('log')
    plt.scatter(medMag2, rms2, marker = '.', s=1, c = '0.2', alpha = 0.025)
    plt.scatter(medMag,rms,s=2,c=validArray,cmap='RdYlGn',alpha=0.5)
    
    
    plt.xlim(12,20)
    plt.ylim(10e-4,1)
    plt.xlabel('Median magnitude [mag]')
    plt.ylabel('rms of magnitude [mag]')
    plt.savefig('MedMagVsRmsBin_Plot',dpi=500, bbox_inches='tight')
    plt.close
    
    
def AppToAbs(appMag, hdu, index):
    dist = hdu[1].data['rest'][np.nonzero(index)]
    return appMag - 5*np.log10(dist) + 5

def CMDIso():
    hdu = fits.open('WFI_GAIA_BJ_NGC3766_MATCH.fits')
    cluster = np.logical_and(hdu[1].data['Cluster1'], hdu[1].data['GAIA_Filter'])
    
    color = hdu[1].data['bp_rp'][cluster]
    mag = hdu[1].data['phot_g_mean_mag'][cluster]
    #mag = AppToAbs(mag, hdu, cluster)
    iso = np.genfromtxt('MIST_iso_5e47f88681046.iso.cmd') # MIST_iso_1Gyr_Av0.iso.cmd MIST_iso_5e47f88681046.iso.cmd
    isoColor = iso[:,23] - iso[:,24] + 0.44   - 0.1
    isoGMag = iso[:,22] + 0.765 + 11.14       + 0.2
    
    plt.gca().invert_yaxis()
    plt.plot(isoColor, isoGMag, c = '0', alpha=0.5)
    plt.scatter(color,mag, s=1, c='0')
    plt.ylabel('Magnitude [mag]')
    plt.xlabel(r'Color ($G_{BP}-G_{RP}$) [mag]')
    plt.ylim(18,10)
    plt.xlim(-0.1,1.75)
    plt.savefig('IsoCMd_NGC3766.png', dpi=200, bbox_inches='tight')
    
    
    
def CreateHistogram(periods, ax=None):
    if ax is None:
        ax = plt.gca()
    return ax.hist(periods, bins = 10, color = '0', histtype = 'step')


def CreateCMDScatter(color, mag, colorScale, ax=None):
    if ax is None:
        ax = plt.gca()
    
    cmd = ax.scatter(color, mag, cmap = 'jet', c = colorScale, s = 5)
    return cmd
    
def CreateIsochrone(isoFile, ax=None):
    if ax is None:
        ax = plt.gca()
        
    iso = np.genfromtxt(isoFile)
    isoColor = iso[:,23] - iso[:,24] + 0.44
    isoGMag = iso[:,22] + 0.765
    return ax.plot(isoColor, isoGMag, color='0', alpha  = 0.5)
    

    
def HistogramCMDAllClusters():
    fig, axs = plt.subplots(3,3, figsize = [12,9], sharex = 'row', sharey = 'row')
    fig.subplots_adjust(top = 1.0, 
                        bottom = 0.01, 
                        hspace=0.3, 
                        wspace=0.1)
    
    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(0,40)
    
    

    # Plotting
    periods2,color2,distance2,absMag2 = FileToData('Period_colour_distance_absMag_M50.txt')
    colorScale = cmap(norm(periods2))
    CreateHistogram(periods2, axs[0,0])
    CreateHistogram(distance2,axs[1,0])
    CreateCMDScatter(color2, absMag2, colorScale, ax = axs[2,0])
    CreateIsochrone('MIST_iso_1Gyr_Av0.iso.cmd', ax = axs[2,0])
    
    
    periods1,color1,distance1,absMag1 = FileToData('Period_colour_distance_absMag_NGC3766_Field.txt')
    colorScale = cmap(norm(periods1))
    CreateHistogram(periods1, axs[0,1])
    CreateHistogram(distance1,axs[1,1])
    CreateCMDScatter(color1, absMag1, colorScale, ax = axs[2,1])
    CreateIsochrone('MIST_iso_1Gyr_Av0.iso.cmd', ax = axs[2,1])
    
    
    
    periods3,color3,distance3,absMag3 = FileToData('Period_colour_distance_absMag_hPer_field.txt')
    delete = [3,10,13,2,9]
    periods3 = np.delete(periods3, delete)
    color3 = np.delete(color3, delete)
    absMag3 = np.delete(absMag3, delete)
    distance3 = np.delete(distance3, delete)
    colorScale = cmap(norm(periods3))
    CreateHistogram(periods3, axs[0,2])
    CreateHistogram(distance3,axs[1,2])
    CreateCMDScatter(color3, absMag3, colorScale, ax = axs[2,2])
    CreateIsochrone('MIST_iso_1Gyr_Av0.iso.cmd', ax = axs[2,2])

    
    # General Setup
    axs[2,0].set_ylim(11,0)
    axs[2,0].set_xlim(0.5,3)
    
    axs[0,0].set_title('NGC2323')
    axs[0,1].set_title('NGC3766')
    axs[0,2].set_title('NGC869')
    
    axs[0,1].set_xlabel('Period [days]')
    axs[1,1].set_xlabel('Distance [pc]')
    axs[2,1].set_xlabel(r'Color ($G_{BP}-G_{RP}$) [mag]')

    
    
    axs[0,0].set_ylabel('Occurance')
    axs[1,0].set_ylabel('Occurance')
    axs[2,0].set_ylabel('Absolute Magnitude [mag]')



    sm =  ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axs, label = 'Period [days]', orientation = 'horizontal', 
                 shrink=0.3, anchor=(0.5,1.6))
    
    plt.savefig('HistogramCMDAllClusters.png', dpi=300, bbox_inches='tight')
    pass
    
    
def MQPeriodColor():
    mQData = fits.open('MQ14_withGaia.fits')
    hdu = fits.open('WFI_GAIA_BJ_NGC3766_MATCH.fits')
    freqs3766 = np.genfromtxt('freqArray3766.txt')
    period50,color50,d,m = FileToData('Period_colour_distance_absMag_M50.txt')
    data869 = np.genfromtxt('a_field_data.txt', delimiter = ',')

    periods3766 = 1/freqs3766[np.nonzero(freqs3766)]
    colour3766 = hdu[1].data['bp_rp'][np.nonzero(freqs3766)]
    
    periodsMQ = mQData[1].data['Prot_MQ14']
    colourMQ = mQData[1].data['bp_rp_g']
    
    plt.figure(figsize = [12,6])
    plt.scatter(colourMQ,periodsMQ, s=0.5, alpha = 0.075, c = '0')
    plt.scatter(color50,period50, s=5, c = 'red', label = 'NGC 2323')
    plt.scatter(colour3766,periods3766, s=5, c = 'green', label = 'NGC 3766')
    plt.scatter(data869[1],data869[0], s=5, c = 'blue', label = 'NGC 869')
    
    plt.ylabel('Period [days]')
    plt.xlabel(r'Color ($G_{BP}-G_{RP}$) [mag]')
    plt.yscale('log')
    plt.ylim(0.1,100)
    plt.yticks([0.1,1,10,100],['0.1','1','10','100'])
    plt.xlim(0.5,2.5)
    red_patch = mpatches.Patch(color='red', label='NGC 2323')
    green_patch = mpatches.Patch(color='green', label='NGC 3766')
    blue_patch = mpatches.Patch(color='blue', label='NGC 869')
    plt.legend(handles=[red_patch,green_patch,blue_patch])
    
    plt.savefig('mcQ_PeriodColourPlot', dpi=200, bbox_inches='tight')

    plt.close
    
def ExLSPOutput():
    hdu = fits.open('WFI_GAIA_BJ_NGC3766_MATCH.fits')
    freqs = np.genfromtxt('freqArray3766.txt')
    index = 5
    indices = np.nonzero(freqs)[0]
    index = indices[index]
    frequency = freqs[index]
    time = hdu[1].data['hjd']
    flux = hdu[1].data['flux']
    
    freq, power = LombScargle(time[index], flux[index]).autopower(minimum_frequency=0.02, 
                                                     maximum_frequency=5,
                                                     samples_per_peak = 15)
    
    
    plt.figure(figsize = (6,3))
    plt.plot(freq, power, c='0', lw = 1)
    plt.ylim(0,1)
    plt.xlim(0,5)
    plt.vlines(frequency,0,1, colors = 'red', lw=5, alpha = 0.5)
    plt.xlabel(r'Frequency [days$^{-1}$]')
    plt.ylabel('LSP Power')
    plt.xticks([0,frequency,1,2,3,4,5],['0',"{:.2f}".format(frequency),'1','2','3','4','5'])
    plt.hlines(0.43,0,5, alpha = 0.5, lw=2, ls = ':')
    plt.savefig('LSPNGC3766Output.png', dpi=200, bbox_inches = 'tight')
    
    
    
    
def GregData():
    data = np.genfromtxt('g_cluster_data.txt', delimiter = ',')
    file = np.stack((data[0], data[1], data[2], data[5]), axis=1)
    np.savetxt('Period_colour_distance_absMag_M50_cluster.txt', file, delimiter = ',')
    
    
def AlexData():
    a_data = fits.open('a_field_data_1.fits')
    
    periods_a = a_data[1].data['periods']
    color_a = a_data[1].data['colour']
    dist_a = a_data[1].data['rest']
    M_G_a = a_data[1].data['phot_g_mean_mag'] + 5 + 5*np.log10(1/dist_a)
    
    file = np.stack((periods_a,color_a,dist_a,M_G_a), axis=1)
    np.savetxt('Period_colour_distance_absMag_hPer_field.txt', file, delimiter = ',')
    
    
SMALL_SIZE = 12
MEDIUM_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels



#AlexData()

#LSUneven()
#TestSunSpot()
#ObsWindows()
#MQPeriodColor()
#PlanckLaw()
#MedMagVsRmsFilter()
#CMDIso()
#HistogramCMDAllClusters()
ExLSPOutput()



