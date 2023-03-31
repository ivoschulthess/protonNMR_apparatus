# imports
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
  
# suppress warnings
warnings.filterwarnings('ignore')

def linFct(x, a, b=0):
    """
    Simple linear function, with or without the linear term. 
    """
    return a*x + b

def ratio (A, B):
    """
    Ratio of two numbers/arrays and its error.
    """
    ratio = A[0] / B[0]
    ratioErr = np.sqrt((A[1]/B[0])**2 + (A[0]*B[1]/B[0]**2)**2)
    
    return np.array([ratio, ratioErr])

def weighted_mean(values, stds, axis=0):
    """
    Weighted mean and error of the mean.
    """
    average = np.average(values, weights=1/stds**2, axis=axis)
    std = 1 / np.sqrt(np.sum(1/stds**2, axis=axis))
        
    return np.array([average, std])

def lorentzAbsFct(f, f0, A, T2, o=0):
    """
    Absorption mode of the Lorentz function/resonance. 
    """
    return A * T2 / (1 + (f-f0)**2 * T2**2) + o

def lorentzDisFct(f, f0, A, T2, o=0):
    """
    Dispersion mode of the Lorentz function/resonance. 
    """
    return -A * (f-f0) * T2**2 / (1 + (f-f0)**2 * T2**2) + o

def lorentzFct(f, f0, A, T2, p, o=0):
    """
    Complex Lorentz function/resonance, including the absorption and dispersion mode.  
    """
    res = lorentzAbsFct(f, f0, A, T2, o) + 1.j*lorentzDisFct(f, f0, A, T2, o)
    res = res*np.exp(-1.j*np.pi*p/180)
    return res

def lorentzFitFct(f, f0, A, T2, p, o=0):
    """
    Helper function to fit the complex Lorentz function/resonance.   
    """
    N = len(f)
    f_real = f[:N//2]
    f_imag = f[N//2:]
    y_real = np.real(lorentzFct(f_real, f0, A, T2, p, o))
    y_imag = np.imag(lorentzFct(f_imag, f0, A, T2, p, o))
    return np.hstack([y_real, y_imag])

def nmrSignal (t, f, a, p, t0):
    """
    FID signal function, a sinusoidal with exponential decay. 
    """
    return a * np.sin(2*np.pi*f*t - p*180/np.pi) * np.exp(-t/t0)

def gaussFct(f, f0, A, s, o):
    """
    Gaussian function, used to fit the Rabi resonances. 
    """
    return A / np.sqrt(2*np.pi*s**2) * np.exp(-(f-f0)**2/(2*s**2)) + o

def sinExpFct (t, f, a, p, o, tau, t0):
    """
    Sinusoidal function with exponential decay, used to fit the Rabi oscillations. 
    """
    return a * np.sin(2*np.pi*f*t - p) * np.exp(-tau*(t-t0)) + o

def sinPhaseFct (t, a, p, o):
    """
    Sinusoidal function with a fixed perios of 360, used to fit the Ramsey phase scans. 
    """
    return abs(a) * np.sin(np.pi/180*(t-p)) + o

def allanFct(x, a, b, c):
    """
    Function to fit the Allan standard deviation. 
    """
    return a / np.sqrt(x) + b*x + c

def rabiFrequencyScan (path, **kwargs):
    """
    Helper function to analyze a Rabi frequency scan. 
    """
    # get line color if in kwargs
    lc = kwargs.get('lc') if 'lc' in kwargs else ''
    
    # get marker size if in kwargs
    ms = float(kwargs.get('ms')) if 'ms' in kwargs else None
        
    # get marker style if in kwargs
    marker = kwargs.get('marker') if 'marker' in kwargs else '.'
    
    # get label if in kwargs
    label = kwargs.get('label') if 'label' in kwargs else ''

    # get initial guess for Gaussian fit if in kwargs
    popt = kwargs.get('p0') if 'p0' in kwargs else (500, -100, 50, 1)
    
    # signal
    data = np.load(path)
    
    F_SF = data['F_SF']
    F_Fit = np.linspace(F_SF[0], F_SF[-1], 1001)
    
    Amp = data['Amp']
    
    popt, pcov = curve_fit(gaussFct, F_SF, Amp[0], sigma=Amp[1], absolute_sigma=True, p0=popt)
    perr = np.sqrt(np.diag(pcov))    
    chi2 = np.sum((Amp[0]-gaussFct(F_SF, *popt))**2 / Amp[1]**2)
    chi2_r = chi2 / (len(Amp[0]) - len(popt))
    
    print('resonance at {:.1f}({:.0f}) Hz'.format(popt[0], 1e1*perr[0]))
    print('resonance width {:.1f}({:.0f}) Hz'.format(abs(popt[2]), 1e1*perr[2]))
    print('FWHM {:.1f}({:.0f}) Hz'.format(2.355*abs(popt[2]), 1e1*2.355*perr[2]))
    print('reduced chi-squared: {:.2f}\n'.format(chi2_r))
        
    if 'ax' in kwargs:
        ax = kwargs.get('ax')
        ax.errorbar(F_SF, Amp[0], Amp[1], fmt='{}{}'.format(lc,marker), ms=ms, lw=1, label=label)
        ax.plot(F_Fit, gaussFct(F_Fit, *popt), '{}-'.format(lc), lw=1)
    
    return popt,perr
    
def rabiAmplitudeScan (path, **kwargs):
    """
    Helper function to analyze a Rabi amplitude scan. 
    """
    # get line color if in kwargs
    lc = kwargs.get('lc') if 'lc' in kwargs else ''
    
    # get marker size if in kwargs
    ms = float(kwargs.get('ms')) if 'ms' in kwargs else None
        
    # get marker style if in kwargs
    marker = kwargs.get('marker') if 'marker' in kwargs else '.'
    
    # get label if in kwargs
    label = kwargs.get('label') if 'label' in kwargs else ''

    # get initial guess for Gaussian fit if in kwargs
    popt = kwargs.get('p0') if 'p0' in kwargs else (0.002,1,0,0.001,100) 
        
    # signal
    data = np.load(path)
    
    A_SF = data['A_SF']
    A_Fit = np.linspace(A_SF[0], 1000, 1001)
    
    Amp = data['Amp']
            
    popt, pcov = curve_fit(sinExpFct, A_SF, Amp[0], sigma=Amp[1], absolute_sigma=True, p0=popt)
    perr = np.sqrt(np.diag(pcov))
    chi2 = np.sum((Amp[0]-sinExpFct(A_SF, *popt))**2 / Amp[1]**2)
    chi2_r = chi2 / (len(Amp[0]) - len(popt))
    
    print('pi/2 flip at {:.1f}({:.0f}) mVpp'.format(1/4/popt[0], 1e1*perr[0]/4/popt[0]**2))
    print('reduced chi-squared: {:.2f}\n'.format(chi2_r))
    
    if 'ax' in kwargs:
        ax = kwargs.get('ax')
        ax.errorbar(A_SF, Amp[0], Amp[1], fmt='{}{}'.format(lc,marker), ms=ms, lw=1, label=label)
        ax.plot(A_Fit, sinExpFct(A_Fit, *popt), '{}-'.format(lc), lw=1)
        
    return popt, perr

def ramseyFrequencyScan (path, **kwargs):
    """
    Helper function to analyze a Ramsey frequency scan. 
    """
    # get line color if in kwargs
    lc = kwargs.get('lc') if 'lc' in kwargs else ''
    
    # get marker size if in kwargs
    ms = float(kwargs.get('ms')) if 'ms' in kwargs else None
        
    # get marker style if in kwargs
    marker = kwargs.get('marker') if 'marker' in kwargs else '.'
    
    # get label if in kwargs
    label = kwargs.get('label') if 'label' in kwargs else ''    
    
    # signal
    data = np.load(path)
    F_SF = data['F_SF']    
    Amp = data['Amp']    
    
    if 'ax' in kwargs:
        ax = kwargs.get('ax')
        ax.errorbar(F_SF, Amp[0], Amp[1], fmt='{}{}'.format(lc,marker), ms=ms, lw=0.5, elinewidth=0.5, label=label)

def ramseyPhaseScan (path, **kwargs):
    """
    Helper function to analyze a Ramsey phase scan. 
    """
    # get line color if in kwargs
    lc = kwargs.get('lc') if 'lc' in kwargs else ''
    
    # get marker size if in kwargs
    ms = float(kwargs.get('ms')) if 'ms' in kwargs else None
        
    # get marker style if in kwargs
    marker = kwargs.get('marker') if 'marker' in kwargs else '.'
    
    # get label if in kwargs
    label = kwargs.get('label') if 'label' in kwargs else ''

    # get initial guess for Gaussian fit if in kwargs
    popt = kwargs.get('p0') if 'p0' in kwargs else (1,0,0) 
    
    # signal
    data = np.load(path)
    
    # calculate the average magnetic field
    By = weighted_mean(data['By'][0], data['By'][1])
    
    P_SF = data['P_SF']
    P_Fit = np.linspace(0, 360, 1001)
    
    Amp = data['Amp']
  
    popt, pcov = curve_fit(sinPhaseFct, P_SF, Amp[0], sigma=Amp[1], absolute_sigma=True, p0=popt)
    perr = np.sqrt(np.diag(pcov))
    chi2 = np.sum((Amp[0]-sinPhaseFct(P_SF, *popt))**2 / Amp[1]**2)
    chi2_r = chi2 / (len(Amp[0]) - len(popt))
    print('\nproton phase: {:.1f}({:.0f}) deg'.format(popt[1], 1e1*perr[1]))
    print('reduced chi-squared: {:.1f}'.format(chi2_r))
    
    if 'ax' in kwargs:
        ax = kwargs.get('ax')
        ax.errorbar(P_SF, Amp[0], Amp[1], fmt='{}{}'.format(lc, marker), ms=ms, lw=1, label=label)
        ax.plot(P_Fit, sinPhaseFct(P_Fit, *popt), '{}-'.format(lc), lw=1)
    
    return popt, perr, By