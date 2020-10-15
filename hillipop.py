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
        self._modenames = ["TT","EE","BB","TE","ET"]
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
        self.dldata = self._read_dl_xspectra( pars['XSpectra'])

        #Weights
        if self.verbose: print( "Read Weights")
        self.dlweight = self._read_dl_xerrors( pars['XSpectraErrors'])

        #Inverted Covariance matrix
        if self.verbose: print( "Read covmat")
        self.invkll = self._read_invcovmatrix( pars['CovMatrix'])

        #Nuisances
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
        Return the (lmin,lmax) for each cross-spectra for each mode (TT,EE,BB,TE)
        array(nmode,nxspec)
        '''
        if self.verbose: print( filename)
        
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
        if self.verbose: print( filename)
        
        dldata = []
        for m1 in range(self.nmap):
            for m2 in range(m1+1,self.nmap):
                tmpcl = []
                #TT EE BB TE ET
                for hdu in [1,2,3,4,4]:
                    data = fits.getdata( "%s_%d_%d.fits" % (filename,m1,m2), hdu)
                    ell = np.array(data.field(0),int)
                    datacl = np.zeros( max(ell)+1)
                    datacl[ell] = data.field(1) * 1e12
                    tmpcl.append( datacl[:self.lmax+1])
                
                dldata.append(tmpcl)
        return( np.transpose( np.array(dldata), ( 1,0,2)))
    
    def _read_dl_xerrors( self, filename):
        '''
        Read xspectra errors from Xpol [Dl in K^2]
        Output: Dl 1/sigma_l^2 in muK^-4
        '''
        if self.verbose: print( filename)
        
        dlweight = []
        for m1 in range(self.nmap):
            for m2 in range(m1+1,self.nmap):
                tmpcl = []
                #TT EE BB TE ET
                for hdu in [1,2,3,4,4]:
                    data = fits.getdata( "%s_%d_%d.fits" % (filename,m1,m2), hdu)
                    ell = np.array(data.field(0),int)
                    datacl = np.zeros( max(ell)+1)
                    datacl[ell] = data.field(2) * 1e12
                    datacl[datacl == 0] = np.inf
                    tmpcl.append( 1./datacl[:self.lmax+1]**2)
                
                dlweight.append(tmpcl)
        return( np.transpose( np.array(dlweight), (1,0,2)))
    
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
        
        #read
        data = fits.getdata( filename+ext+".fits").field(0)
        nel = int(np.sqrt(len(data)))
        data = data.reshape( (nel,nel))/1e24  #muK^-4
        
        if nel != nell:
            raise ValueError('Incoherent covariance matrix')
        
        return( data)
    
    def _select_spectra( self, cl, mode=0):
        '''
        Cut spectra given Multipole Ranges and flatten
        '''
        acl = np.asarray(cl)
        xl = []
        for xf in range(self.nxfreq):
            lmin = self.lmins[mode][self.xspec2xfreq.index(xf)]
            lmax = self.lmaxs[mode][self.xspec2xfreq.index(xf)]
            xl = xl+list(acl[xf,lmin:lmax+1])
        return( np.array(xl))
    
    def _xspectra_to_xfreq( self, cl, weight):
        '''
        Average cross-spectra per cross-frequency
        '''
        xcl = np.zeros( (self.nxfreq, self.lmax+1))
        xw8 = np.zeros( (self.nxfreq, self.lmax+1))
        for xs in range(self.nxspec):
            xcl[self.xspec2xfreq[xs]] += weight[xs] * cl[xs]
            xw8[self.xspec2xfreq[xs]] += weight[xs]
        
        xw8[xw8 == 0] = np.inf
        return( xcl / xw8)
    
    def _compute_residuals( self, pars, cl_boltz):
        #cl_boltz from Boltzmann (Cl in K^2)
        lth = np.arange( self.lmax+1)
        dlth = np.asarray(cl_boltz)[:,lth] * (lth*(lth+1)/2./np.pi * 1e12) #Dl in muK^2
        
        #nuisances
        cal = []
        for m1 in range(self.nmap):
            for m2 in range(m1+1,self.nmap):
                cal.append( pars["Aplanck"]*pars["Aplanck"] * (1.+ pars["c%d" % m1] + pars["c%d" % m2]))
        
        #TT
        if self.isTT:
            dlmodel = [dlth[0]]*self.nxspec
            for fg in self.fgsTT:
                dlmodel += fg.compute_dl( pars)
            
            #Compute Rl = Dl - Dlth
            Rspec = [self.dldata[0][xs] - cal[xs]*dlmodel[xs] for xs in range(self.nxspec)]
            Rl = self._xspectra_to_xfreq( Rspec, self.dlweight[0])
        
        return( Rl)
    
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

        chi2 = Xl.dot(self.invkll).dot(Xl)

        return( chi2)

#------------------------------------------------------------------------------------------------

