#
#HILLIPOP
#
#Sep 2020   - M. Tristram -
import numpy as np
from numpy.linalg import *
import astropy.io.fits as fits
import scipy.ndimage as nd

tagnames = ['TT','EE','BB','TE','TB','EB']

from foregrounds_v3 import *
from tools import *



class hillipop(object):
    """
    High-L Likelihood for Polarized Planck
    Spectra-based Gaussian-approximated likelihood with foreground models for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency maps
    """
    def __init__( self, paramfile, verbose=False):
        '''
        init Hillipop likelihood
        
        Parameters
        ----------
        paramfile: string
            parameter file containing likelihood information
        '''
        
        pars = read_parameter( paramfile)
        self.verbose = verbose
        self._modenames = ["TT","EE","TE","ET"]
        self.nmap = int(pars["map"])
        self._set_modes( pars)
        self.fqs = self._get_freqs( pars)
        self.nfreq = len(np.unique(self.fqs))
        self.nxfreq = self.nfreq*(self.nfreq+1)/2
        self.nxspec = self.nmap*(self.nmap-1)/2
        self._set_lists()
        
        #Multipole ranges
        if self.verbose: print( "Multipole Ranges")
        self._set_multipole_ranges( pars['MultipolesRange'])        

        #Data
        if self.verbose: print( "Read Data")
        self.dldata = self._read_dl_xspectra( pars['XSpectra'], field=1)

        #Weights
        if self.verbose: print( "Read Weights")
        dlsig = self._read_dl_xspectra( pars['XSpectraErrors'], field=2)
        dlsig[dlsig == 0] = np.inf
        self.dlweight = 1./dlsig**2

        #Inverted Covariance matrix
        if self.verbose: print( "Read covmat")
        self.invkll = self._read_invcovmatrix( pars['CovMatrix'])

        #Nuisances
        self.parname = ["Aplanck"]
        for m in range(self.nmap): self.parname.append( "c%d" % m)

        #Foregrounds
        self.fgs = [] #list of foregrounds per mode [TT,EE,TE,ET]
        #Init foregrounds TT
        fgsTT = []
        if self.isTT:
            fgsTT.append( ps_radio( self.lmax, self.fqs, self.parname))
            fgsTT.append( ps_dusty( self.lmax, self.fqs, self.parname))
            if "Dust" in pars.keys():
                fgsTT.append( dust_model( self.lmax, self.fqs, self.parname, pars["Dust"], mode="TT"))
            if "SZ" in pars.keys():
                fgsTT.append( sz_model( self.lmax, self.fqs, self.parname, pars["SZ"]))
            if "CIB" in pars.keys():
                fgsTT.append( cib_model( self.lmax, self.fqs, self.parname, pars["CIB"]))
            if "kSZ" in pars.keys():
                fgsTT.append( ksz_model( self.lmax, self.fqs, self.parname, pars["kSZ"]))
            if "SZxCIB" in pars.keys():
                fgsTT.append( szxcib_model( self.lmax, self.fqs, self.parname, pars["SZxCIB"]))
        self.fgs.append(fgsTT)
    
        #Init foregrounds EE
        fgsEE = []
        if self.isEE:
            if "Dust" in pars.keys():
                fgsEE.append( dust_model( self.lmax, self.fqs, self.parname, pars["Dust"], mode="EE"))
        self.fgs.append(fgsEE)

        #Init foregrounds TE
        fgsTE = []
        if self.isTE:
            if "Dust" in pars.keys():
                fgsTE.append( dust_model( self.lmax, self.fqs, self.parname, pars["Dust"], mode="TE"))
        self.fgs.append(fgsTE)
        fgsET = []
        if self.isET:
            if "Dust" in pars.keys():
                fgsET.append( dust_model( self.lmax, self.fqs, self.parname, pars["Dust"], mode="ET"))
        self.fgs.append(fgsET)

    def _set_modes(self,pars):
        self.isTT = ( int(pars["TT"]) == 1 )
        self.isEE = ( int(pars["EE"]) == 1 )
        self.isTE = ( int(pars["TE"]) == 1 )
        self.isET = ( int(pars["ET"]) == 1 )
        self._is_mode = [self.isTT,self.isEE,self.isTE or self.isET]
    
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
        
        list_fqs = []
        for f1 in range(self.nfreq):
            for f2 in range(f1, self.nfreq):
                list_fqs.append( (f1,f2))
        
        freqs = list(np.unique(self.fqs))
        self.xspec2xfreq = []
        for m1 in range(self.nmap):
            for m2 in range(m1+1,self.nmap):
                f1 = freqs.index(self.fqs[m1])
                f2 = freqs.index(self.fqs[m2])
                self.xspec2xfreq.append( list_fqs.index((f1,f2)))
    
    def _set_multipole_ranges( self, filename):
        '''
        Return the (lmin,lmax) for each cross-spectra for each mode (TT,EE,TE,ET)
        array(nmode,nxspec)
        '''
        if self.verbose: print( filename)
        
        self.lmins = []
        self.lmaxs = []
        for hdu in [0,1,3,4]:  #file HDU [TT,EE,BB,TE,ET]
            data = fits.getdata( filename, hdu+1)
            self.lmins.append( np.array(data.field(0),int))
            self.lmaxs.append( np.array(data.field(1),int))
        self.lmax = max([max(l) for l in self.lmaxs])
    
    def _read_dl_xspectra( self, filename, field=1):
        '''
        Read xspectra from Xpol [Dl in K^2]
        Output: Dl in muK^2
        '''
        if self.verbose: print( filename)
        
        dldata = []
        for m1 in range(self.nmap):
            for m2 in range(m1+1,self.nmap):
                tmpcl = []
                for hdu in [1,2,4]:   #TT EE BB TE
                    data = fits.getdata( "%s_%d_%d.fits" % (filename,m1,m2), hdu)
                    ell = np.array(data.field(0),int)
                    datacl = np.zeros( max(ell)+1)
                    datacl[ell] = data.field(field) * 1e12
                    tmpcl.append( datacl[:self.lmax+1])
                
                data = fits.getdata( "%s_%d_%d.fits" % (filename,m2,m1), hdu)
                ell = np.array(data.field(0),int)
                datacl = np.zeros( max(ell)+1)
                datacl[ell] = data.field(field) * 1e12
                tmpcl.append( datacl[:self.lmax+1])
                
                dldata.append(tmpcl)
        
        return( np.transpose( np.array(dldata), ( 1,0,2)))    
    
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
        if self.verbose: print( filename+ext+".fits")
        
        #count dim
        nell = 0
        for m in range(3):
            if self._is_mode[m]:
                nells = self.lmaxs[m]-self.lmins[m]+1
                nell += sum([nells[self.xspec2xfreq.index(k)] for k in range(self.nxfreq)])
        
        #read
        data = fits.getdata( filename+ext+".fits").field(0)
        nel = int(np.sqrt(len(data)))
        data = data.reshape( (nel,nel))/1e24  #muK^-4
        
        if nel != nell:
            raise ValueError('Incoherent covariance matrix (read:%d, expected:%d)' % (nel,nell))
        
        return( data)
    
    def _select_spectra( self, cl, mode=0):
        '''
        Cut spectra given Multipole Ranges and flatten
        Return: list
        '''
        acl = np.asarray(cl)
        xl = []
        for xf in range(self.nxfreq):
            lmin = self.lmins[mode][self.xspec2xfreq.index(xf)]
            lmax = self.lmaxs[mode][self.xspec2xfreq.index(xf)]
            xl = xl+list(acl[xf,lmin:lmax+1])
        return( xl)
    
    def _xspectra_to_xfreq( self, cl, weight, normed=True):
        '''
        Average cross-spectra per cross-frequency
        '''
        xcl = np.zeros( (self.nxfreq, self.lmax+1))
        xw8 = np.zeros( (self.nxfreq, self.lmax+1))
        for xs in range(self.nxspec):
            xcl[self.xspec2xfreq[xs]] += weight[xs] * cl[xs]
            xw8[self.xspec2xfreq[xs]] += weight[xs]
        
        xw8[xw8 == 0] = np.inf
        if normed:
            return( xcl / xw8)
        else:
            return( xcl, xw8)
    
    def _compute_residuals( self, pars, dlth, mode=0):
        
        #nuisances
        cal = []
        for m1 in range(self.nmap):
            for m2 in range(m1+1,self.nmap):
                cal.append( pars["Aplanck"]*pars["Aplanck"] * (1.+ pars["c%d" % m1] + pars["c%d" % m2]))
        
        #Data
        dldata = self.dldata[mode]

        #Model
        dlmodel = [dlth[mode]]*self.nxspec
        for fg in self.fgs[mode]:
            dlmodel += fg.compute_dl( pars)
        
        #Compute Rl = Dl - Dlth
        Rspec = np.array([dldata[xs] - cal[xs]*dlmodel[xs] for xs in range(self.nxspec)])
        
        return( Rspec)
    
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
        clth = np.asarray( cl_boltz)
        lth = np.arange( self.lmax+1)
        imodes = [0,1,3,3] #TT, EE, TE, ET
        dlth = np.asarray(clth[imodes])[:,lth] * (lth*(lth+1)/2./np.pi * 1e12) #Dl in muK^2
        
        #Create Data Vector
        Xl = []
        if self.isTT:
            #compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals( pars, dlth, mode=0)
            #average to cross-spectra
            Rl = self._xspectra_to_xfreq( Rspec, self.dlweight[0])
            #select multipole range
            Xl = Xl + self._select_spectra( Rl, mode=0)
        
        if self.isEE:
            #compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals( pars, dlth, mode=1)
            #average to cross-spectra
            Rl = self._xspectra_to_xfreq( Rspec, self.dlweight[1])
            #select multipole range
            Xl = Xl + self._select_spectra( Rl, mode=1)
        
        if self.isTE or self.isET:
            #compute residuals Rl = Dl - Dlth
            if self.isTE:
                Rspec = self._compute_residuals( pars, dlth, mode=2)
                RlTE,WlTE = self._xspectra_to_xfreq( Rspec, self.dlweight[2], normed=False)
            if self.isET:
                Rspec = self._compute_residuals( pars, dlth, mode=3)
                RlET,WlET = self._xspectra_to_xfreq( Rspec, self.dlweight[3], normed=False)                
            if self.isTE:
                if self.isET:
                    Rl = (RlTE+RlET) / (WlTE + WlET)
                else:
                    Rl = RlTE/WlTE
            elif self.isET:
                Rl = RlET/WlET
            #select multipole range
            Xl = Xl + self._select_spectra( Rl, mode=2)
        
        Xl = np.array(Xl)
        chi2 = Xl.dot(self.invkll).dot(Xl)
        
        return( chi2)

#------------------------------------------------------------------------------------------------

