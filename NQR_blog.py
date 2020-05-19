# -*- coding: utf-8 -*-
"""
Code to accompany:
    
Simulation of Nuclear Quadrupole Resonance Spectra of
14N and 2H in Phenylalanine Using an NV Center Magnetometer

Bhaskar Abhiraman

Instructions
-Run cells individually
-Bulk calculations take a while


"""
#%% 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import linalg as LA
from astropy.modeling.models import Lorentz1D
from matplotlib.widgets import Slider, Button, RadioButtons


#Take hbar to be 1
pi = np.pi

plt.close('all')
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16)
plt.rc('axes', labelsize=16) 




def spectra_B(B0, th, phi, QCC, eta):
    mub = 2398203.6
    g = 2 
    Bx = B0*np.sin(th)*np.cos(phi)
    By = B0*np.sin(th)*np.sin(phi)
    Bz = B0*np.cos(phi)
    Sz = np.asarray([[1, 0, 0],
                 [0, 0, 0],
                 [0, 0, -1]])
    Sx = (1/np.sqrt(2))*np.asarray([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]])
    Sy = (1/np.sqrt(2))*np.asarray([[0, -1j, 0],
                                    [1j, 0, -1j],
                                    [0, 1j, 0]])
    Hq = (QCC/4) *(3*Sz.dot(Sz) + eta*(Sx.dot(Sx) - Sy.dot(Sy)))
    Hz = -g*mub*(Sx*Bx + Sy*By + Sz*Bz)
    eig_es = LA.eigvalsh(Hq + Hz - (QCC/2)*np.identity(3))
    diffs = [eig_es[2]-eig_es[1], eig_es[1]-eig_es[0], eig_es[2]-eig_es[0]]
    return eig_es, np.abs(diffs)


def deltifier(peaks, start, stop, res):
    fs = np.arange(start, stop+res, res)
    amp = np.zeros_like(fs)
    for peak in peaks:
        ind = np.argmin(np.abs(fs - peak))
        amp[ind] = 1
    return fs, amp

def spectra_Gen(ths, phis, Bs, QCCs, etas, f_0, f_end, f_res, fudge = 1):
    #Bs_D = np.linspace(0,0.02,100) #was 100
    fs_D = np.zeros((3,len(Bs),3))
    eigs_D = np.zeros((3,len(Bs),3))
    #f_0, f_end, f_res = 3.26768e6 - 2e6, 3.26768e6 + 2e6, 1e3
    fs = np.arange(f_0, f_end+f_res, f_res)
    Lor = Lorentz1D(amplitude = 1, x_0 = (f_0+f_end)/2, fwhm = 4e3)(fs)
    num_f = len(np.arange(f_0, f_end+f_res, f_res))
    spectra = np.zeros((num_f,len(Bs)))
    
    for k in range(len(ths)):
        th = ths[k]
        phi = phis[k]
        for j, QCC_D in enumerate(QCCs):
            for i, B in enumerate(Bs):
                eigs_D[:,i,j], fs_D[:,i,j] = spectra_B(fudge*B, th, phi, QCC_D, etas[j])
                #print('Progress: {} %'.format(100*(i+1)/len(Bs_D)))
              
        for i, B in enumerate(Bs):
            peaks = np.ndarray.flatten(fs_D[:,i,:])
            _, amp = deltifier(peaks, f_0, f_end, f_res)
            spectra[:,i] += np.convolve(amp, Lor, 'same')   

    return fs, Bs, spectra

def spectra_Gen_bulk(Bs, QCCs, etas, f_0, f_end, f_res, fudge = 1):
    fs_D = np.zeros((3,len(Bs),3))
    eigs_D = np.zeros((3,len(Bs),3))
    #f_0, f_end, f_res = (0e3, 200e3, 1e3)
    fs = np.arange(f_0, f_end+f_res, f_res)
    Lor = Lorentz1D(amplitude = 1, x_0 = (f_0+f_end)/2, fwhm = 4e3)(fs)
    num_f = len(np.arange(f_0, f_end+f_res, f_res))
    spectra = np.zeros((num_f,len(Bs)))
    ths = np.linspace(0, pi, 100)
    phis = np.linspace(0, 2*pi, 200)
    cnt = 0
    for k in range(len(ths)):
        th = ths[k]
        for l in range(len(phis)):
            cnt +=1
            phi = phis[l]
            print('Bulk progress: {} %'.format(100*(cnt)/(len(phis)*len(ths))))
            for j, QCC_D in enumerate(QCCs):
                for i, B in enumerate(Bs):
                    eigs_D[:,i,j], fs_D[:,i,j] = spectra_B(fudge*B, th, phi, QCC_D, etas[j])
                    #print('Progress: {} %'.format(100*(i+1)/len(Bs_D)))                  
            for i, B in enumerate(Bs):
                peaks = np.ndarray.flatten(fs_D[:,i,:])
                _, amp = deltifier(peaks, f_0, f_end, f_res)
                spectra[:,i] += np.sin(th)*np.convolve(amp, Lor, 'same')   

    return fs, Bs, spectra

def spectra_N_angle(ths, phis):
    B = 0.25
    Bs_D = [B]
    QCCs = [1.354e6]
    etas = [0.63]
    fs_D = np.zeros((3,len(Bs_D),3))
    eigs_D = np.zeros((3,len(Bs_D),3))
    f_0, f_end, f_res = (0e5, 50e5, 5e3)
    fs = np.arange(f_0, f_end+f_res, f_res)
    Lor = Lorentz1D(amplitude = 1, x_0 = (f_0+f_end)/2, fwhm = 50e3)(fs)
    num_f = len(np.arange(f_0, f_end+f_res, f_res))
    spectra = np.zeros((num_f,len(ths)))
    
    for k in range(len(ths)):
        th = ths[k]
        phi = phis[k]
        eigs_D[:,0,0], fs_D[:,0,0] = spectra_B(B, th, phi, QCCs[0], etas[0])
        #print('Progress: {} %'.format(100*(i+1)/len(Bs_D)))
        peaks = np.ndarray.flatten(fs_D[:,0,:])
        _, amp = deltifier(peaks, f_0, f_end, f_res)
        spectra[:,k] += np.convolve(amp, Lor, 'same')   

    return fs, Bs_D, spectra

def spectra_Gen_angle(ths, phis, B, QCCs, etas,f_0, f_end, f_res, fudge = 1, fwhm = 4e3):
    Bs = [B]
    fs_D = np.zeros((3,len(Bs),3))
    eigs_D = np.zeros((3,len(Bs),3))
    fs = np.arange(f_0, f_end+f_res, f_res)
    Lor = Lorentz1D(amplitude = 1, x_0 = (f_0+f_end)/2, fwhm = fwhm)(fs)
    num_f = len(np.arange(f_0, f_end+f_res, f_res))
    spectra = np.zeros((num_f,len(ths)))
    
    for i in range(len(QCCs)):
        for k in range(len(ths)):
            th = ths[k]
            phi = phis[k]
            eigs_D[:,0,i], fs_D[:,0,i] = spectra_B(fudge*B, th, phi, QCCs[i], etas[i])
            #print('Progress: {} %'.format(100*(i+1)/len(Bs_D)))
            peaks = np.ndarray.flatten(fs_D[:,0,i])
            _, amp = deltifier(peaks, f_0, f_end, f_res)
            spectra[:,k] += np.convolve(amp, Lor, 'same')   

    return fs, Bs, spectra

def draw_spec(fs, Bs_D, spec, ax):
    im = ax.pcolormesh(fs/1e3,Bs_D*1e3, spec.T, cmap = 'jet')
    ax.margins(x=0)
    #cb = fig.colorbar(im, ax = ax)
    ax.set_xlabel('NMR frequency (kHz)')
    ax.set_ylabel('Magnetic field (mT)')

# Deuterium two orientation, toggle thetas
def spec_gui_angles(ths, phis, Bs, QCCs, etas, f_0, f_end, f_res):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True,figsize = (9,9))

    ax.margins(x = 0)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    
    
    axcolor = 'lightgray'
    axphi1 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axth1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    sphi1 = Slider(axphi1, r'$\phi_1$', 0, pi, valinit=phi1_i, valstep=0.1)
    sth1 = Slider(axth1, r'$\theta_1$', 0, 2*pi, valinit=th1_i)
    axphi2 = plt.axes([0.25, 0.0, 0.65, 0.03], facecolor=axcolor)
    axth2 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    sphi2 = Slider(axphi2, r'$\phi_2$', 0, pi, valinit=phi2_i, valstep=0.1)
    sth2 = Slider(axth2, r'$\theta_2$', 0, 2*pi, valinit=th2_i)
    
    def update(val):
        ths[0] = sth1.val
        phis[0] = sphi1.val
        ths[1] = sth2.val
        phis[1] = sphi2.val
        #print(th1,phi1,th2,phi2)
        fs, Bs_D, spec = spectra_Gen(ths, phis, Bs, QCCs, etas, f_0, f_end, f_res)
        draw_spec(fs, Bs_D, spec, ax)
        #l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        #fig.canvas.draw_idle()
    for slider in (sphi1,sth1,sphi2,sth2):
        slider.on_changed(update)
        slider.on_changed(update)
    
    fs, Bs_D, spec = spectra_Gen(ths, phis,Bs, QCCs, etas, f_0, f_end, f_res)
    draw_spec(fs, Bs_D, spec, ax)

def spectra_Gen_angle_sweep(B, QCCs, etas, f0, f_end, f_res, fudge = 1, fwhm = 4e3):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True,figsize = (7,6),)
    n = 400
    fs_na, _, spec_na_th = spectra_Gen_angle(np.linspace(0,pi,n),n*[0],B, QCCs, etas,f_0, f_end, f_res, fudge = fudge, fwhm = fwhm)
    _, _, spec_na_ph = spectra_Gen_angle(n*[0],np.linspace(0,pi,n),B, QCCs, etas,f_0, f_end, f_res, fudge = fudge, fwhm = fwhm)
    spec_angle = np.hstack((spec_na_th, spec_na_ph))
    #angs = np.hstack([np.linspace(0,pi,n)[::-1], np.linspace(0,pi,n)])
    im_na = ax.pcolormesh(fs_na[10:801]/13, np.linspace(0,pi,2*n), spec_angle[10:801,:].T, cmap = 'jet')
    ax.set_yticks([])
    ax.set_xlabel('NMR frequency (kHz)')
    fig.colorbar(im_na)
    return fig, ax


#%% TWO ORIENTATIONS OF DEUTERIUM
Bs_D = np.linspace(0,0.02,100)
QCCs_D = [120e3, 110e3, 130e3]
etas_D = [0,0,0]
f_0, f_end, f_res = (0e3, 200e3, 1e3)
th1_i, phi1_i, th2_i, phi2_i = (0.27, 0.4, 1.1, 2.50)

spec_gui_angles([th1_i, th2_i], [phi1_i, phi2_i], Bs_D, QCCs_D, etas_D, f_0, f_end, f_res)

#%% DEUTERIUM BULK SWEEP (Takes a long time)
Bs_D = np.linspace(0,0.02,30) #was 100
QCCs_D = [120e3, 110e3, 130e3]
etas_D = [0,0,0]
f_0, f_end, f_res = (0e3, 200e3, 10e3)
fs_db, bs_db, spec_db = spectra_Gen_bulk(Bs_D, QCCs_D, etas_D, f_0, f_end, f_res)
figdb, axdb = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True,figsize = (6,6))
draw_spec(fs_db, bs_db, spec_db, axdb)
#%% N BULK FOR ONE B VALUE (Takes a while)
QCCs_N = [1.354e6]
etas_N = [0.63]
f_0, f_end, f_res = (0e5, 50e5, 1e4)
fs_NB, Bs_NB, spec_NB = spectra_Gen_bulk([0.5], QCCs_N, etas_N, f_0, f_end, f_res, fudge = 0.5)
fs_nblk = np.arange(0, 5e6+1e4, 1e4)
figb, axb = plt.subplots(nrows = 1, ncols = 1, constrained_layout = True,figsize = (9,3))
axb.plot(fs_NB[10:-10]/1e6, spec_NB[10:-10], lw = 3)
axb.set_xlabel('NMR frequency (MHz)')
axb.set_ylabel('NMR contrast (a.u.)')
axb.set_yticks([])
axb.set_title('N-14 Bulk spectrum for B = 0.5 T')


#%% N angle sweep
QCCs_N = [1.354e6]
etas_N = [0.63]
f_0, f_end, f_res = (0e5, 50e5, 1e4)
fig_na, ax_na = spectra_Gen_angle_sweep(0.5, QCCs_N, etas_N, f_0, f_end, f_res, fudge = 0.5, fwhm = 50e3)
ax_na.set_title('N-14 sweep for B = 0.5 T')

#%% Deut angle sweep
QCCs_D = [120e3, 110e3, 130e3]
etas_D = [0,0,0]
f_0, f_end, f_res = (0e3, 200e3, 1e3)
fig_da, ax_da = spectra_Gen_angle_sweep(0.01, QCCs_D, etas_D, f_0, f_end, f_res)
ax_da.set_title('Deut sweep for B = 10 mT')
