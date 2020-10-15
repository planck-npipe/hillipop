#FOREGROUNDS V3 (Planck 2018)
import numpy as np
from numpy.linalg import *
import astropy.io.fits as fits
import scipy.ndimage as nd

from foregrounds import fgmodel


#Radio Point Sources
class ps_radio( fgmodel):
    def __init__( self, lmax, freqs, parname, auto=False):
        #Amplitudes of the point sources power spectrum (MJy^2.sr-1) per xfreq
        #(100x100,100x143,100x217,143x143,143x217,217x217)
        self.refq = [100,143,217]
        self.radio = [[7.76, 5.36, 4.26],
                      [5.36, 4.83, 3.60],
                      [4.26, 3.60, 3.22]]
        self.lmax = lmax
        self.freqs = freqs
        self._shift = 0 if auto else 1
        self.name = "PS radio"
        parname.append('Aradio')
        ell = np.arange( self.lmax+1)
        self.dl_ps = ell*(ell+1)/2./np.pi

    def compute_dl( self, pars):
        nfreq = len(self.freqs)
        dl_radio = []
        for f1 in range(nfreq):
            for f2 in range(f1+self._shift,nfreq):
                freq1 = self.freqs[f1]
                freq2 = self.freqs[f2]
                dl_radio.append( self.dl_ps * pars['Aradio'] * self.radio[self.refq.index(freq1)][self.refq.index(freq2)] / self.gnu[freq1] / self.gnu[freq2])
        return(np.array(dl_radio))


#Infrared Point Sources
class ps_dusty( fgmodel):
    def __init__( self, lmax, freqs, parname, auto=False):
        #Amplitudes of the point sources power spectrum (MJy^2.sr-1) per xfreq
        #(100x100,100x143,100x217,143x143,143x217,217x217)
        self.refq = [100,143,217]
        self.dusty = [[0.18179145, 0.47110054,  1.90295795],
                      [0.47110054, 1.24375538,  5.06409993],
                      [1.90295795, 5.06409993, 20.99528025]]
        self.lmax = lmax
        self.freqs = freqs
        self._shift = 0 if auto else 1
        self.name = "PS dusty"
        parname.append('Adusty')
        ell = np.arange( self.lmax+1)
        self.dl_ps = ell*(ell+1)/2./np.pi

    def compute_dl( self, pars):
        nfreq = len(self.freqs)
        dl_dusty = []
        for f1 in range(nfreq):
            for f2 in range(f1+self._shift,nfreq):
                freq1 = self.freqs[f1]
                freq2 = self.freqs[f2]
                dl_dusty.append(self.dl_ps * pars['Adusty'] * self.dusty[self.refq.index(freq1)][self.refq.index(freq2)] / self.gnu[freq1] / self.gnu[freq2])
        return(np.array(dl_dusty))

#Dust model
class dust_model( fgmodel):
    def __init__( self, lmax, freqs, parname, filename, mode='TT', auto=False):
        _hnu_T = {100:0.01957,143:0.03982,217:0.13185}
        _hnu_P = {100:0.01703,143:0.03605,217:0.12498}
        self.lmax = lmax
        self.mode = mode
        self._shift = 0 if auto else 1
        self.name = "Dust"
        self.mode = mode
        if mode == 'TT': parname.append('AdustTT')
        if mode == 'EE': parname.append('AdustPP')
        if mode == 'TE': parname.append('AdustTP')
        if mode == 'ET': parname.append('AdustTP')

        nfreq = len(freqs)
        if self.mode == 'TT':
            icol = 1
            hnuA = hnuB = _hnu_T
        if self.mode == 'EE':
            icol = 2
            hnuA = hnuB = _hnu_P
        if self.mode == 'TE':
            icol = 4
            hnuA = _hnu_T
            hnuB = _hnu_P
        if self.mode == 'ET':
            icol = 5
            hnuA = _hnu_P
            hnuB = _hnu_T
        self.dl_dust = []
        for f1 in range(nfreq):
            for f2 in range(f1+self._shift,nfreq):
                dust_in = fits.getdata( "%s_%dx%d.fits" % (filename,freqs[f1],freqs[f2]))
                ell = np.array(dust_in.field(0),int)
                tmpl = np.zeros( max(ell)+1)
                tmpl[ell] = ell*(ell+1)/2./np.pi * hnuA[freqs[f1]]*hnuB[freqs[f2]]*dust_in.field(icol)
                self.dl_dust.append(tmpl[:self.lmax+1])

    def compute_dl( self, pars):            
        if self.mode == 'TT': Ad = pars['AdustTT']
        if self.mode == 'EE': Ad = pars['AdustPP']
        if self.mode == 'TE': Ad = pars['AdustTP']
        if self.mode == 'ET': Ad = pars['AdustTP']        
        return( Ad * np.array(self.dl_dust))


#tSZ (one spectrum for all freqs)
class sz_model( fgmodel):
    def __init__( self, lmax, freqs, parname, filename, auto=False):
        #template: Dl=l(l+1)/2pi Cl, units uK at 143GHz
        self.lmax = lmax
        self._shift = 0 if auto else 1
        self.name = "tSZ"
        parname.append('Asz')
        nfreq = len(freqs)
        self.dl_sz = []
        dldata = fits.getdata( filename)
        ell = np.array(dldata.ELL,int)
        for f1 in range(nfreq):
            for f2 in range(f1+self._shift,nfreq):
                tmpl = np.zeros( max(ell)+1)
                tmpl[ell] = (dldata.CL1HALO+dldata.CL2HALO)*self.fnu[freqs[f1]]*self.fnu[freqs[f2]]/self.fnu[143]/self.fnu[143]
                self.dl_sz.append(tmpl[:lmax+1])
    
    def compute_dl( self, pars):
        return(pars['Asz'] * np.array(self.dl_sz))


#CIB model (one spectrum per xfreq)
class cib_model( fgmodel):
    def __init__( self, lmax, freqs, parname, filename, auto=False):
        self.lmax = lmax
        self._shift = 0 if auto else 1
        self.name = "CIB"
        parname.append('Acib')
        nfreq = len(freqs)
        self.dl_cib = []
        for f1 in range(nfreq):
            for f2 in range(f1+self._shift,nfreq):
                cldata = fits.getdata( "%s_%dx%d.fits" % (filename,freqs[f1],freqs[f2]))
                ell = np.array( cldata.ELL, int)
                tmpl = np.zeros( max(ell)+1)
                ll2pi = ell*(ell+1)/2./np.pi
                tmpl[ell] = ll2pi*(cldata.CL1HALO+cldata.CL2HALO)/self.gnu[freqs[f1]]/self.gnu[freqs[f2]]
                self.dl_cib.append(tmpl[:lmax+1])
    
    def compute_dl( self, pars):
        return(pars['Acib'] * np.array(self.dl_cib))


#kSZ (one spectrum for all freqs)
class ksz_model( fgmodel):
    def __init__( self, lmax, freqs, parname, filename, auto=False):
        #template: Dl=l(l+1)/2pi Cl, units uK
        self.lmax = lmax
        self._shift = 0 if auto else 1
        self.name = "kSZ"
        parname.append('Aksz')
        nfreq = len(freqs)
        self.dl_ksz = []
        dltemp = fits.getdata( filename)
        ell = np.array( dltemp.ELL, int)
        for f1 in range(nfreq):
            for f2 in range(f1+self._shift,nfreq):
                tmp = np.zeros( max(ell)+1)
                tmp[ell] = dltemp.DELL
                self.dl_ksz.append(tmp[:lmax+1])
    
    def compute_dl( self, pars):
        return(pars['Aksz'] * np.array(self.dl_ksz))


#SZxCIB model (one spectrum per xfreq)
class szxcib_model( fgmodel):
    def __init__( self, lmax, freqs, parname, filename, auto=False):
        self.lmax = lmax
        self._shift = 0 if auto else 1
        self.name = "SZxCIB"
        parname.append('Aszxcib')
        nfreq = len(freqs)
        self.dl_szxcib = []
        for f1 in range(nfreq):
            for f2 in range(f1+self._shift,nfreq):
                cldata = fits.getdata( "%s_%dx%d.fits" % (filename,freqs[f1],freqs[f2]))
                ell = np.array( cldata.ELL, int)
                tmpl = np.zeros( max(ell)+1)
                ll2pi = ell*(ell+1)/2./np.pi
                tmpl[ell] = ll2pi*cldata.CL/self.gnu[freqs[f1]]/self.gnu[freqs[f2]]
                self.dl_szxcib.append(tmpl[:lmax+1])
    
    def compute_dl( self, pars):
        return(pars['Aszxcib'] * np.array(self.dl_szxcib))





