#
#HILLIPOP tools library
#
#Jun 2016   - M. Tristram -
import numpy as np
#import mpfit as mp
from numpy.linalg import *
import astropy.io.fits as fits
import scipy.ndimage as nd
import matplotlib.pyplot as plt
#import pymc
#import xpol as xp

tagnames = ['TT','EE','BB','TE','TB','EB']






#------------------------------------------------------------------------------------------------
#Hillipop Gaussian Likelihood (No Foregrounds)
#------------------------------------------------------------------------------------------------
class likelihood(object):
    """
    Hillipop full-multipole
    """
    def __init__( self, cldata, fullkll, lrange):
        '''
        init and compute invert covariance matrix
        
        * cldata: array(nbin,ncross)
                  data array of spectra
        * fullkll: array(nl*ncross,nl*ncross)
                   covariance matrix
        * lrange: (lmin,lmax)
        '''
        self.CMB = True
        
        #cldata[nl,ncross]
        self.lmin,self.lmax = lrange
        self.nel = self.lmax-self.lmin+1
        
        #Data
        self.cldata = cldata
        bla,self.ncross = np.shape(cldata)
        
        #Kll
        kll = cut_kll( fullkll, self.ncross, (self.lmin,self.lmax))
        self.invkll = inv(kll)
    
    def compute_likelihood( self, cl_boltz):
        '''
        Compute likelihood from model out of Boltzmann code
        Units: Cl in K^2
        '''
        #cl_boltz from Boltzmann (Cl in K^2)
        lmin,lmax,ncross,nel = self.lmin,self.lmax,self.ncross,self.nel
        lth = np.arange( 0, lmax+1)
        
        #Compute Dlth
        clth = np.asarray(cl_boltz[lth]) * (lth*(lth+1)/2./np.pi)
        
        #Compute Rl = Dl - Dlth
        Xl = np.reshape([self.cldata[lmin:lmax+1,c]-clth[lmin:lmax+1] for c in range(ncross)],ncross*nel)
        
        return( np.dot( np.dot( Xl, self.invkll), Xl))


class bin_likelihood(object):
    """
    Hillipop binned-likelihood
    """
    #likelihood in Cl
    def __init__( self, cbdata, kbb, bintab):
        '''
        init and compute invert covariance matrix
        
        * cldata: array(nbin,ncross)
                  data array of spectra
        * kbb: array(nbin*ncross,nbin*ncross)
               covariance matrix
        *bintab: Bins class (Xpol)
        '''
        self.CMB = True
        self.bintab = bintab
        
        #Data
        self.cbdata = cbdata
        self.ncross,self.nbins = np.shape(cbdata)
        if self.nbins != bintab.nbins:
            raise ValueError('Binning not coherent')
        
        #Kbb
        self.invkbb = np.linalg.inv(kbb)
    
        
    def compute_likelihood( self, cl_boltz):
        '''
        Compute binned-likelihood from model out of Boltzmann code
        Units: Cl in K^2
        '''
        #bin Clth
        cbth = self.bintab.bin_spectra( cl_boltz)
        
        #Compute Rb = Cb - Cbth
        Xb = np.reshape([self.cbdata[c]-cbth for c in range(self.ncross)],self.ncross*self.nbins)
        
        return( np.dot( np.dot( Xb, self.invkbb), Xb))
    
    def compute_model( self, pico, parpico):
        cl = hlp_model( pico, parpico)
        return( self.compute_likelihood(cl))
#------------------------------------------------------------------------------------------------






#------------------------------------------------------------------------------------------------
#Hillipop with foregounds
#------------------------------------------------------------------------------------------------
class hillipop(object):
    """
    High-L Likelihood for Polarized Planck
    Spectra-based Gaussian-approximated likelihood with foreground models for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency maps
    """
    def __init__( self, paramfile):
        '''
        init Hillipop likelihood
        
        Parameters
        ----------
        paramfile: string
            parameter file containing likelihood information
        '''

        pars = read_parameter( paramfile)
        self.CMB = True
        self._modenames = ["TT","EE","BB","TE","ET"]
        self.nmap = int(pars["map"])
        self._set_modes( pars)
        self.fqs = self._get_freqs( pars)
        self.nfreq = len(np.unique(self.fqs))
        self.nxfreq = self.nfreq*(self.nfreq+1)/2
        self.nxspec = self.nmap*(self.nmap-1)/2
        self._set_lists()
        
        #Multipole ranges
        print( "Multipole Ranges")
        self._set_multipole_ranges( pars['MultipolesRange'])        
        #Data
        print( "Read Data")
        self.dldata = self._read_dl_xspectra( pars['XSpectra'])

        #Weights
        print( "Read Weights")
        self.dlweight = self._read_dl_xerrors( pars['XSpectraErrors'])

        #Inverted Covariance matrix
        print( "Read covmat")
        self.invkll = self._read_invcovmatrix( pars['CovMatrix'])

        self.parname = ["Aplanck"]
        for m in range(self.nmap): self.parname.append( "c%d" % m)
        #Init foregrounds TT
        if self.isTT:
            self.fgsTT = []
            self.fgsTT.append( ps_radio( self.lmax, self.fqs, self.parname))
            self.fgsTT.append( ps_dusty( self.lmax, self.fqs, self.parname))
            if "Dust" in pars.keys():
                self.fgsTT.append( dust_model( self.lmax, self.fqs, self.parname, pars["Dust"], mode="TT"))
            if "SZ" in pars.keys():
                self.fgsTT.append( sz_model( self.lmax, self.fqs, self.parname, pars["SZ"]))
            if "CIB" in pars.keys():
                self.fgsTT.append( cib_model( self.lmax, self.fqs, self.parname, pars["CIB"]))
            if "kSZ" in pars.keys():
                self.fgsTT.append( ksz_model( self.lmax, self.fqs, self.parname, pars["kSZ"]))
            if "SZxCIB" in pars.keys():
                self.fgsTT.append( szxcib_model( self.lmax, self.fqs, self.parname, pars["SZxCIB"]))
    
        #Init foregrounds EE
        if self.isEE:
            self.fgsEE = []
            if "Dust" in pars.keys():
                self.fgsEE.append( dust_model( self.lmax, self.fqs, self.parname, pars["Dust"], mode="EE"))

        #Init foregrounds TE
        if self.isTE:
            self.fgsTE = []
            if "Dust" in pars.keys():
                self.fgsTE.append( dust_model( self.lmax, self.fqs, self.parname, pars["Dust"], mode="TE"))
        if self.isET:
            self.fgsET = []
            if "Dust" in pars.keys():
                self.fgsET.append( dust_model( self.lmax, self.fqs, self.parname, pars["Dust"], mode="ET"))


    def _set_modes(self,pars):
        self.isTT = True if int(pars["TT"]) == 1 else False
        self.isEE = True if int(pars["EE"]) == 1 else False
        self.isTE = True if int(pars["TE"]) == 1 else False
        self.isET = True if int(pars["ET"]) == 1 else False
    
    def _get_freqs(self,pars):
        fqs = []
        for f in range(self.nmap):
            fqs.append( int(pars["freq%d"%f]))
        return( fqs)

    def _set_lists(self):
        self.xspec2map = []
        list_xfq = []
        for m1 in range(self.nmap):
            for m2 in range(m1+1,self.nmap):
                self.xspec2map.append( (m1,m2))
                list_xfq.append( self.fqs[m1]*self.fqs[m2])
        list_xfq = list(np.unique(list_xfq))

        nxspec = len(self.xspec2map)
        self.xspec2xfreq = []
        for m1 in range(self.nmap):
            for m2 in range(m1+1,self.nmap):
                self.xspec2xfreq.append( list_xfq.index(self.fqs[m1]*self.fqs[m2]))
    
    def _set_multipole_ranges( self, filename):
        '''
        Return the (lmin,lmax) for each cross-spectra for each mode (TT,EE,BB,TE)
        array(nmode,nxspec)
        '''
        self.lmins = []
        self.lmaxs = []
        for m in range(4):
            data = fits.getdata( filename, m+1)
            self.lmins.append( np.array(data.field(0),int))
            self.lmaxs.append( np.array(data.field(1),int))
        self.lmax = max([max(l) for l in self.lmaxs])
    
    def _read_dl_xspectra( self, filename):
        '''
        Read xspectra from Xpol [Dl in K^2]
        Output: Dl in muK^2
        '''
        dldata = []
        for m1 in range(self.nmap):
            for m2 in range(m1+1,self.nmap):
                tmpcl = []
                #TT EE BB TE
                for m in range(4):
                    data = fits.getdata( "%s_%d_%d.fits" % (filename,m1,m2), m)
                    ell = np.array(data.field(0),int)
                    datacl = np.zeros( max(ell)+1)
                    datacl[ell] = data.field(1) * 1e12
                    tmpcl.append( datacl[:self.lmax+1])
                #ET
                data = fits.getdata( "%s_%d_%d.fits" % (filename,m2,m1), 4)
                ell = np.array(data.field(0),int)
                datacl = np.zeros( max(ell)+1)
                datacl[ell] = data.field(1) * 1e12
                tmpcl.append( datacl[:self.lmax+1])
                
                dldata.append(tmpcl)
        return( np.transpose( dldata, ( 1,0,2)))
                
    def _read_dl_xerrors( self, filename):
        '''
        Read xspectra errors from Xpol [Dl in K^2]
        Output: Dl 1/sigma_l^2 in muK^-4
        '''
        dlweight = []
        for m1 in range(self.nmap):
            for m2 in range(m1+1,self.nmap):
                tmpcl = []
                #TT EE BB TE
                for m in range(4):
                    data = fits.getdata( "%s_%d_%d.fits" % (filename,m1,m2), m)
                    ell = np.array(data.field(0),int)
                    datacl = np.zeros( max(ell)+1)
                    datacl[ell] = data.field(1) * 1e12
                    tmpcl.append( 1./datacl[:self.lmax+1]**2)
                #ET
                data = fits.getdata( "%s_%d_%d.fits" % (filename,m2,m1), 4)
                ell = np.array(data.field(0),int)
                datacl = np.zeros( max(ell)+1)
                datacl[ell] = data.field(1) * 1e12
                tmpcl.append( 1./datacl[:self.lmax+1]**2)

                dlweight.append(tmpcl)
        return( np.transpose( dlweight, (1,0,2)))
    
    def _read_invcovmatrix( self, filename):        
        '''
        Read xspectra inverse covmatrix from Xpol [Dl in K^-4]
        Output: invkll [Dl in muK^-4]
        '''
        ext = "_"
        if self.isTT: ext += "TT"
        if self.isEE: ext += "EE"
        if self.isTE: ext += "TE"
        if self.isET: ext += "ET"
        print( filename+ext+".fits")

        #count dim
        nell = 0
        if self.isTT:
            nells = self.lmaxs[0]-self.lmins[0]+1
            nell += sum([nells[self.xspec2xfreq.index(k)] for k in range(self.nxfreq)])
        if self.isEE:
            nells = self.lmaxs[1]-self.lmins[1]+1
            nell += sum([nells[self.xspec2xfreq.index(k)] for k in range(self.nxfreq)])
        if self.isTE:
            nells = self.lmaxs[2]-self.lmins[2]+1
            nell += sum([nells[self.xspec2xfreq.index(k)] for k in range(self.nxfreq)])
        if self.isET:
            nells = self.lmaxs[3]-self.lmins[3]+1
            nell += sum([nells[self.xspec2xfreq.index(k)] for k in range(self.nxfreq)])
        
        data = fits.getdata( filename+ext+".fits").field(0)
        nel = int(np.sqrt(len(data)))
        data = data.reshape( nel,nel)/1e24
        
        if nel != nell:
            raise ValueError('Incoherent covariance matrix')

        return( data)

    def _select_spectra( self, cl, mode=0):
        '''
        Cut spectra given Multipole Ranges and flatten
        '''
        xl = []
        for xf in range(self.nxfreq):
            lmin = self.lmins[mode][self.xspec2xfreq.index(xf)]
            lmax = self.lmaxs[mode][self.xspec2xfreq.index(xf)]
            xl = xl+list(cl[xf,lmin:lmax+1])
        return( np.array(xl))

    def _xspectra_to_xfreq( self, cl, weight):
        '''
        Average cross-spectra per cross-frequency
        '''
        xcl = np.zeros( (self.nxfreq, self.lmax+1))
        xw8 = np.zeros( (self.nxfreq, self.lmax+1))
        k = 0
        for xs in range(self.nxspec):
            xcl[self.xspec2xfreq[xs]] += weight[xs] * cl[xs]
            xw8[self.xspec2xfreq[xs]] += weight[xs]
        
        return( xcl / xw8)
    
    def compute_likelihood( self, pars, cl_boltz):
        '''
        Compute likelihood from model out of Boltzmann code
        Units: Dl in muK^2

        Parameters
        ----------
        pars: dict
              parameter values
        cl_boltz: array or arr2d
              CMB power spectrum (Cl in K^2)

        Returns
        -------
        lnL: float
            Log likelihood for the given parameters -2ln(L)
        '''
        #cl_boltz from Boltzmann (Cl in K^2)
        lth = np.arange( self.lmax+1)
        dlth = np.asarray(cl_boltz[:,lth]) * (lth*(lth+1)/2./np.pi * 1e12) #Dl in muK^2
        
        #nuisances
        cal = []
        for m1 in range(self.nmap):
            for m2 in range(m1+1,self.nmap):
                cal.append( pars["Aplanck"]*pars["Aplanck"] * (1.+pars["c%d" % m1]) * (1.+pars["c%d" % m2]))
        
        #TT
        if self.isTT:
            dlmodel = [dlth[0]]*self.nxspec
            for fg in self.fgsTT:
                dlmodel += fg.compute_dl( pars)
            
            #Compute Rl = Dl - Dlth
            Rl = self._xspectra_to_xfreq( [self.dldata[0][xs] - cal[xs]*dlmodel[xs] for xs in range(self.nxspec)], self.dlweight[0])
            Xl = self._select_spectra( Rl)
        
        return( np.dot( np.dot( Xl, self.invkll), Xl))

#------------------------------------------------------------------------------------------------






#------------------------------------------------------------------------------------------------
#Foreground classes
#------------------------------------------------------------------------------------------------
class fgmodel(object):
    '''
    Class of foreground model for the Hillipop likelihood
    Units: Dl in muK^2
    '''
    
    # MJy.sr-1 -> Kcmb for (100,143,217) GHz
    gnu = {100:244.059, 143:371.658, 217:483.485}
    #frequency dependence (100,143,217,353,545,857) GHz of the SZ effect
    fnu = {100:-4.031, 143:-2.785, 217:0.187, 353:6.205, 545:14.455, 857:26.335}

    def __init__( self, freqs, parname, filename, auto=False):
        '''
        Create model for foreground
        '''
        self.freq = freqs
        self.name = "MyFg"
        parname.append( "par1")
        pass
    
    def compute_dl( self, pars):
        '''
        Return spectra model for each cross-frequency
        '''
        pass



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
                tmpl[ell] = ll2pi*(cldata.CL1HALO+cldata.CL1HALO)/self.gnu[freqs[f1]]/self.gnu[freqs[f2]]
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









def hlp_fgmodel( pars):
    # MJy.sr-1 -> Kcmb for (100,143,217) GHz
    freq = [100,143,217]
    nfreq = len(nfreq)
    gnu = [244.059, 371.658, 483.485]
    _gnu = []
    for f1 in range(nfreq):
        for f2 in range(f1,nfreq):
            _gnu.append( gnu[f1]*gnu[f2])
    nxfreq = len(_gnu)
    
    #Point sources
    #Amplitudes of the point sources power spectrum (MJy^2.sr-1) per xfreq (100x100,100x143,100x217,143x143,143x217,217x217)
    radio = [7.76, 5.36, 4.26, 4.83, 3.6, 3.22]
    dusty = [0.18179145, 0.47110054, 1.90295795, 1.24375538, 5.06409993, 20.99528025]
    cl_radio = radio * gnuxfq
    cl_dusty = dusty * gnuxfq
    
    #Dust (one spectrum per xfreq)
    _hnu_T = [0.01957,0.03982,0.13185]
    _hnu_P = [0.01703,0.03605,0.12498]
    cl_dust_tt = []
    cl_dust_ee = []
    cl_dust_te = []
    cl_dust_et = []
    for f1 in range(nfreq):
        for f2 in range(f1,nfreq):
            dust_in = fits.getdata( dustfile, 1)
            tmpl = np.zeros( lmax+1)
            tmpl[dust_in.field(0)] = _hnu_T[f1]*_hnu_T[f2]*dust_in.field(1)
            cl_dust_tt.append(tmpl)
            
            dust_in = fits.getdata( dustfile, 2)
            tmpl = np.zeros( lmax+1)
            tmpl[dust_in.field(0)] = _hnu_P[f1]*_hnu_P[f2]*dust_in.field(1)
            cl_dust_ee.append(tmpl)
            
            dust_in = fits.getdata( dustfile, 4)
            tmpl = np.zeros( lmax+1)
            tmpl[dust_in.field(0)] = _hnu_T[f1]*_hnu_P[f2]*dust_in.field(1)
            cl_dust_te.append(tmpl)
            
            dust_in = fits.getdata( dustfile, 5)
            tmpl = np.zeros( lmax+1)
            tmpl[dust_in.field(0)] = _hnu_P[f1]*_hnu_T[f2]*dust_in.field(1)
            cl_dust_te.append(tmpl)
    pars['AdustTT'] = 1.
    pars['AdustTP'] = 1.
    pars['AdustPP'] = 1.
    
    #tSZ
    #frequency dependence (100,143,217,353,545,857) GHz of the SZ effect
    fnu = [-4.031, -2.785, 0.187, 6.205, 14.455, 26.335]
    #template: Dl=l(l+1)/2pi Cl, units uK at 143GHz
    szinput = fits.getdata( szfile)
    cl_sz = []
    for f1 in range(nfreq):
        for f2 in range(f1,nfreq):
            tmpl = np.zeros( lmax+1)
            tmpl[szinput.ell] = 2.*pi/szinput.ell/(szinput.ell+1)*(szinput.Cl1haloSZ+szinput.Cl2haloSZ)*fnu[f1]*fnu[f2]/fnu[1]/fnu[1]
            cl_sz.append(tmpl)
    pars['Asz'] = 1.
    
    #CIB (one spectrum per xfreq)
    cl_cib = []
    for f1 in range(nfreq):
        for f2 in range(f1,nfreq):
            cib_in = fits.getdata( )
            tmpl = np.zeros( lmax+1)
            tmpl[cib_in.field(0)] = (cib_in.field(1)+cib_in.field(2))/gnu[f1]/gnu[f2]
            cl_cib.append(tmpl)
    pars['Acib'] = 1.
    
    #kSZ (one spectrum)
    #template: Dl=l(l+1)/2pi Cl, units uK
    ksz_in = fits.getdata( kszfile)
    cl_ksz = []
    for f1 in range(nfreq):
        for f2 in range(f1,nfreq):
            tmp = np.zeros( lmax+1)
            tmp[ksz_in.field(0)] = ksz_in.field(1) / (ksz_in.field(0)*(ksz_in.field(0)+1)/2./pi)
            cl_ksz.append(tmp)
    pars['Aksz'] = 1.
    
    #SZxCIB
    cl_szxcib = []
    for f1 in range(nfreq):
        for f2 in range(f1,nfreq):
            szxicb_in = fits.getdata( szxcibfile)
            tmpl = np.zeros( lmax+1)
            tmpl[szxcib_in.field(0)] = szxcib_in.field(1)/_gnu[f][0]/_gnu[f][1]
            cl_szxcib.append(tmpl)
    pars['Aszxcib'] = 1.
    
#------------------------------------------------------------------------------------------------






#------------------------------------------------------------------------------------------------
#PRIOR
#------------------------------------------------------------------------------------------------
class prior:
    from scipy.interpolate import interp1d
    from scipy.interpolate import InterpolatedUnivariateSpline
    
    def __init__(self, parname):
        self.CMB = False
        self.parname = parname
#        pass

    def gauss( self, mean, sigma):
        self._pdf = lambda x: np.exp( -0.5*((x-mean)/sigma)**2)
    
    def tabulated( self, val, lik, order=2):
        self._pdf = self.InterpolatedUnivariateSpline( val, lik, k=order)
#        self._pdf = self.interp1d( val, lik, kind='quadratic')
    
    def compute_likelihood( self, value):
        if isinstance( value, dict):
            l = self._pdf(value[self.parname])
        else:
            l = self._pdf(value)
        
        return( -2.*np.log(l))

#------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------
#Tools
#------------------------------------------------------------------------------------------------
def read_parameter( filename):
    d = {}
    FILE = open(filename)
    for line in FILE:
        name, value = line.split("=")
        value = value.strip()
        if " " in value:
            value = map(str, value.split())
        else:
            value = str(value)
        d[name.strip()] = value
#        setattr(self, d[name], value)
    return(d)

def list_cross( nmap):
    return( [(i,j) for i in range(0,nmap) for j in range(i+1,nmap)])


def read_kll( dir, clname, icross, tag, lmax=300):
    nl = (lmax+1) - 2  #first 2 multipoles not written
    ncross = len(icross)
    
    kll = np.zeros( (nl*ncross,nl*ncross))
    for c1 in range(ncross):
        cname1 = "%d%d" % icross[c1]
        for c2 in range(ncross):
            cname2 = "%d%d" % icross[c2]
            kllfile = "%s/kll/kll_%s_%s_%s.fits" % (dir,clname,cname1,cname2)
            print( kllfile)
            data = fits.getdata( kllfile)
            nel = int(np.sqrt(len(data))/5)
            data = data.field(0).reshape( (nel*5, nel*5))
            kll[nl*c1:nl*c1+nl,nl*c2:nl*c2+nl] = data[tag*nel:tag*nel+nl,tag*nel:tag*nel+nl]
            kll[nl*c2:nl*c2+nl,nl*c1:nl*c1+nl] = data[tag*nel:tag*nel+nl,tag*nel:tag*nel+nl].T
    
    return( kll)

def read_cls( dir, clname, icross, tag, shift=0):
    ncross = len(icross)
    lmax = 10000
    
    cldata = np.zeros( (lmax+1, ncross))
    for c in range(ncross):
        file1 = 'cross_%s_%d_%d.fits' % (clname,icross[c][0],icross[c][1]+shift)
        file2 = 'cross_%s_%d_%d.fits' % (clname,icross[c][0]+shift,icross[c][1])
        print(file1,file2)
        xpol1 = fits.getdata( '%s/spectra/%s' % (dir,file1), tag+1)
        xpol2 = fits.getdata( '%s/spectra/%s' % (dir,file2), tag+1)
        cldata[np.array(xpol1.ell,int),c] = (xpol1.cell + xpol2.cell)/2.
    
    lmax = int(np.max(xpol1.ell))
    return( np.resize( cldata, (lmax+1, ncross)))


def cut_kll( fullkll, ncross, lrange):
    lmin,lmax = lrange
    nnel = lmax-lmin+1

    fnel = len(fullkll)/ncross
    kll = np.zeros( (nnel*ncross, nnel*ncross))
    for c1 in range(ncross):
        for c2 in range(ncross):
            kll[c1*nnel:(c1+1)*nnel,c2*nnel:(c2+1)*nnel] = fullkll[c1*fnel+lmin-2:c1*fnel+lmax-1,c2*fnel+lmin-2:c2*fnel+lmax-1]

    return(kll)




def QUtoEB( map, nside):
    alms = hp.map2alm( map, lmax=3*nside-1, pol=True)
    mapE = hp.alm2map( alms[1], nside, lmax=3*nside-1, pol=False)
    mapB = hp.alm2map( alms[2], nside, lmax=3*nside-1, pol=False)
    return( mapE, mapB)



def create_bin_file( filename, lbinTT, lbinEE, lbinBB, lbinTE, lbinET):
    """
    lbin = [(lmin,lmax)] for each 15 cross-spectra
    """
    h = fits.Header()
    hdu = [fits.PrimaryHDU(header=h)]

    def fits_layer( lbin):
        h = fits.Header()
        lmin = np.array([l[0] for l in lbin])
        lmax = np.array([l[1] for l in lbin])
        c1 = fits.Column(name='LMIN', array=lmin, format='1D')
        c2 = fits.Column(name='LMAX', array=lmax, format='1D')
        return(fits.BinTableHDU.from_columns([c1,c2],header=h))

    hdu.append(fits_layer( lbinTT))
    hdu.append(fits_layer( lbinEE))
    hdu.append(fits_layer( lbinBB))
    hdu.append(fits_layer( lbinTE))
    hdu.append(fits_layer( lbinET))

    hdulist = fits.HDUList(hdu)
    hdulist.writeto( filename, overwrite=True)



#smooth cls before Cov computation
def SG( l, cl, nsm=5, lcut=0):
    clSG = np.copy(cl)
    
    #gauss filter
    if lcut < 2*nsm:
        shift=0
    else:
        shift=2*nsm
    
    data = nd.gaussian_filter1d( clSG[max(0,lcut-shift):], nsm)
    clSG[lcut:] = data[shift:]
    
    return clSG


def posterior2D( xvals, yvals, allchi2, parnames, cmaps=None, xlim=None, ylim=None, names=None, alpha=0.8):
    nele = len(allchi2)

    if cmaps is None:
        cmaps = ["Blues","Greens","Reds","Greys"]

    plt.figure(figsize=(6,6))
    for it,chi2 in enumerate(np.asarray(allchi2)):
        chi2[np.isnan(chi2)] = np.inf

        L = np.array(np.exp( -0.5*(chi2-chi2.min()))).T
        cmap = plt.cm.get_cmap(cmaps[it])
        
        ax=plt.subplot( 2, 2, 3)
        lvls=np.append(list(ctr_level( L, [0.68,0.95])[::-1]),L.max())
        plt.contourf( xvals, yvals, L, levels=lvls,colors=None, extent=(xvals.min(),xvals.max(),-1,1), cmap=cmap, alpha=alpha)
        plt.xlabel( parnames[0])
        plt.ylabel( parnames[1])
        ax.locator_params(tight=True, nbins=6)
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        
        ax=plt.subplot( 2, 2, 4)
        L1 = np.sum( L,1)
        plt.plot( yvals, L1/np.max(L1), color=cmap(0.7))
#        plt.plot( yvals, L[:,np.argmin( np.abs(xvals))]/np.sum(L[:,np.argmin( np.abs(xvals))]), color=cmap(0.5), ls='--')
        ax.yaxis.set_visible(False)
        ax.locator_params(tight=True, nbins=6)
        plt.xlabel( parnames[1])
        if ylim is not None: plt.xlim(ylim)
        
        ax=plt.subplot( 2, 2, 1)
        L0 = np.sum( L, 0)
        plt.plot( xvals, L0/np.max(L0), color=cmap(0.7))
#        plt.plot( xvals, L[np.argmin( np.abs(yvals)),:]/np.sum(L[np.argmin( np.abs(yvals)),:]), color=cmap(0.5), ls='--')
        ax.locator_params(tight=True, nbins=6)
        ax.yaxis.set_visible(False)
        if xlim is not None: plt.xlim(xlim)

    if names is not None:
        ax=plt.subplot( 2, 2, 1)
        ax.legend( names, loc='upper center', bbox_to_anchor=(1.5, 0.5), prop={'size': 10})


def convert_to_stdev(sigma):
    """
    Given a grid of likelihood values, convert them to cumulative
    standard deviation.  This is useful for drawing contours from a
    grid of likelihoods.
    """
#    sigma = np.exp(-logL+np.max(logL))

    shape = sigma.shape
    sigma = sigma.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(sigma)[::-1]
    i_unsort = np.argsort(i_sort)

    sigma_cumsum = sigma[i_sort].cumsum()
    sigma_cumsum /= sigma_cumsum[-1]
    
    return sigma_cumsum[i_unsort].reshape(shape)


def ctr_level(histo2d, lvl):
    """
    Extract the contours for the 2d plots
    """
    
    h = histo2d.flatten()*1.
    h.sort()
    cum_h = np.cumsum(h[::-1])
    cum_h /= cum_h[-1]
    
    alvl = np.searchsorted(cum_h, lvl)
    clist = h[-alvl]
    
    return clist



def BinSpectra( l, cl, ecl=[], dl=40, l2=True):

    fact = 1.
    if l2:
        fact = l*(l+1)/2./np.pi

    if ecl != []:
        el = ecl
    else:
        el = np.ones( len(cl))

    #apply fact
    cl = cl*fact
    el = el*fact

    lb = []
    cb = []
    eb = []
    for i in np.arange( min(l), max(l), dl):
        w8 = np.sum(1./el[i:i+dl]**2)
        lb.append( i+dl/2)
        cb.append( np.sum( cl[i:i+dl]/el[i:i+dl]**2)/w8)
        if ecl != []:
            eb.append( np.sqrt( 1./w8))
        else:
            eb.append( np.std( cl[i:i+dl])/np.sqrt(dl))

    return( (lb,cb,eb))

#------------------------------------------------------------------------------------------------






#--------------------------------------------------------------------------------------------------
#pyMC
#------------------------------------------------------------------------------------------------
plck_params = { 'ombh2': 0.02212,
                'omch2': 0.1210,
                'tau': 0.058,
                'As': 3.058,
                'ns': 0.9649,
                'theta': 1.04164}

def hlp_model( pars, pico):
    '''
    retrieve PICO model in Cl
    '''
    dl = pico.get(force=True, outputs=['cl_TT'], **pars)['cl_TT'] #l(l+1)/2/np.pi*cl
    
    ell = np.arange(len(dl))
    fact = ell*(ell+1)/2/np.pi
    
    cl = np.zeros(len(dl))
    cl[2:] = dl[2:]*1e-12/fact[2:]
    
    return( cl)

##PICO: pico3_tailmonty_v34.dat
## 0.01016 < re_optical_depth < 0.18221
## 0.059113 < omch2 < 0.20703
## 1.7992e-09 < scalar_amp(1) < 3.3973e-09
## 0.01017 < theta < 0.010617
## 0.70769 < scalar_spectral_index(1) < 1.2556
## 0.017588 < ombh2 < 0.026751
def cosmo_model( datasets, variables, pico, fidvalues=plck_params, clmodel=hlp_model):
    if not isinstance(datasets, list): datasets=[datasets]
    ombh2 = pymc.Uniform('ombh2', 0.018, 0.0267, value=plck_params['ombh2'], observed=('ombh2' not in variables))
    omch2 = pymc.Uniform('omch2', 0.08, 0.16, value=plck_params['omch2'], observed=('omch2' not in variables))
    theta = pymc.Uniform('theta', 1.02, 1.06, value=plck_params['theta'], observed=('theta' not in variables))
    tau   = pymc.Uniform('tau',   0.04, 0.10, value=plck_params['tau']  , observed=('tau'   not in variables))
    As    = pymc.Uniform('As',    2.9 , 3.5 , value=plck_params['As']   , observed=('As'    not in variables))
    ns    = pymc.Uniform('ns',    0.8 , 1.2 , value=plck_params['ns']   , observed=('ns'    not in variables))
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood( value=0., ombh2=ombh2, omch2=omch2, tau=tau, As=As, ns=ns, theta=theta, pico=pico, clmodel=clmodel):
        ll=0.
        parpico = pico.example_inputs().copy()
        parpico['pivot_scalar'] = 0.05
        parpico['ombh2'] = ombh2
        parpico['omch2'] = omch2
        parpico['re_optical_depth'] = tau
        parpico['scalar_amp(1)'] = np.exp(As)/1e10
        parpico['scalar_spectral_index(1)'] = ns
        parpico['theta'] = theta/100.
        cl = clmodel( parpico, pico)
        for ds in datasets:
            if ds.CMB:
                ll = ll + (-0.5*ds.compute_likelihood(cl))
            else:
                ll = ll + (-0.5*ds.compute_likelihood(parpico))
        return(ll)
    return(locals())


def run_mcmc(data, pico, variables=['ombh2','omch2','theta','tau','As','ns'], model=cosmo_model,
             delay=1000, niter=80000, nburn=20000, nthin=1):

    mcmodel = model(data, variables, pico, fidvalues=plck_params)
    
    proposal = {'ombh2':0.05,'omch2':0.05,'theta':0.05,'tau':0.05,'As':0.05,'ns':0.05}
    scales = {}
    for v in variables:
        scales[mcmodel[v]] = proposal[v]
    
    chain = pymc.MCMC( mcmodel)
    
    chain.use_step_method(pymc.AdaptiveMetropolis,chain.stochastics,scales=scales,delay=delay)
    chain.sample(iter=niter,burn=nburn,thin=nthin)
    
    ch ={}
    for v in variables: ch[v] = chain.trace(v)[:]
    
    return ch



class Bins(object):
    """
        lmins : list of integers
            Lower bound of the bins
        lmaxs : list of integers
            Upper bound of the bins
    """
    def __init__( self, lmins, lmaxs):
        if not(len(lmins) == len(lmaxs)):
            raise ValueError('Incoherent inputs')

        lmins = np.asarray( lmins)
        lmaxs = np.asarray( lmaxs)
        cutfirst = np.logical_and(lmaxs>=2 ,lmins>=2)
        self.lmins = lmins[cutfirst]
        self.lmaxs = lmaxs[cutfirst]
        
        self._derive_ext()
    
    @classmethod
    def fromdeltal( cls, lmin, lmax, delta_ell):
        nbins = (lmax - lmin + 1) // delta_ell
        lmins = lmin + np.arange(nbins) * delta_ell
        lmaxs = lmins + delta_ell-1
        return( cls( lmins, lmaxs))

    def _derive_ext( self):
        for l1,l2 in zip(self.lmins,self.lmaxs):
            if( l1>l2):
                raise ValueError( "Incoherent inputs")
        self.lmin = min(self.lmins)
        self.lmax = max(self.lmaxs)
        if self.lmin < 1:
            raise ValueError('Input lmin is less than 1.')
        if self.lmax < self.lmin:
            raise ValueError('Input lmax is less than lmin.')
        
        self.nbins = len(self.lmins)
        self.lbin = (self.lmins + self.lmaxs) / 2.
        self.dl   = (self.lmaxs - self.lmins + 1)

    def bins(self):
        return (self.lmins,self.lmaxs)
    
    def cut_binning(self, lmin, lmax):
        sel = np.where( (self.lmins >= lmin) & (self.lmaxs <= lmax) )[0]
        self.lmins = self.lmins[sel]
        self.lmaxs = self.lmaxs[sel]
        self._derive_ext()
    
    def _bin_operators(self,Dl=False,cov=False):
        if Dl:
            ell2 = np.arange(self.lmax+1)
            ell2 = ell2 * (ell2 + 1) / (2 * np.pi)
        else:
            ell2 = np.ones(self.lmax+1)
        p = np.zeros((self.nbins, self.lmax+1))
        q = np.zeros((self.lmax+1, self.nbins))
        
        for b, (a, z) in enumerate(zip(self.lmins, self.lmaxs)):
            dl = (z-a+1)
            p[b, a:z+1] = ell2[a:z+1] / dl
            if cov:
                q[a:z+1, b] = 1 / ell2[a:z+1] / dl
            else:
                q[a:z+1, b] = 1 / ell2[a:z+1]
        
        return p, q

    def bin_spectra(self, spectra, Dl=False):
        """
        Average spectra in bins specified by lmin, lmax and delta_ell,
        weighted by `l(l+1)/2pi`.
        Return Cb
        """
        spectra = np.asarray(spectra)
        minlmax = min([spectra.shape[-1] - 1,self.lmax])
#        if Dl:
#            fact_binned = 1.
#        else:
#            fact_binned = 2 * np.pi / (self.lbin * (self.lbin + 1))
        
        _p, _q = self._bin_operators()
        return np.dot(spectra[..., :minlmax+1], _p.T[:minlmax+1,...]) #* fact_binned


