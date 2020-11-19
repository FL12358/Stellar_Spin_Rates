import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits

MAXPERIOD = 50

def CheckForMeasured(pIn, pOut, powers):
    th = 0.1 # proportion difference
    minPower = 0.43 # LSP power
    
    Measured = np.where(powers > minPower)[0]
    NotMeasured = np.where(powers <= minPower)[0]
    Correct = np.where(abs(pIn-pOut)<=th*pIn)[0]
    Incorrect = np.where(abs(pIn-pOut)>th*pIn)[0]
    
    return Measured, NotMeasured, Correct, Incorrect
    

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





hdu = fits.open('WFI_NCG3766_GAIA_Match1.fits')

constantId = np.genfromtxt('WFI_Constant_star_Indices', dtype = 'int32')
pIn = np.genfromtxt('Input_Periods.txt')
pOut = np.genfromtxt('Output_Periods.txt')
powers = np.genfromtxt('Powers.txt')

PeriodCompletenessPlot(pIn, pOut, powers)
PeriodReliabiltyContaiminationPlot(pIn, pOut, powers)



'''
testID = 13
testP = 40
time, flux = CreateSyntheticLC(testID, hdu, testP)
t1 = np.arange(0,60,0.01)
f1 = np.sin(2*np.pi*t1/testP)
plt.scatter(t1,f1,s=1)
plt.scatter(time, flux-flux[0], s=5)
'''




