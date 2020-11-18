#------------------------------------------------------------------------------------------------
#Hillipop Tools
#------------------------------------------------------------------------------------------------
import os
import numpy as np
from numpy.linalg import *
import astropy.io.fits as fits
import scipy.ndimage as nd

from hillipop import foregrounds_v3 as fg


tagnames = ["TT","EE","TE","ET"]


# ------------------------------------------------------------------------------------------------
# Likelihood
# ------------------------------------------------------------------------------------------------

class hlp_likelihood():
    """High-L Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency
    maps

    """

    def __init__(self, multipoles_range_file, xspectra_basename, xerrors_basename, covariance_matrix_basename,
                 frequencies, foregrounds, log, data_folder = "",
                 TT=False, EE=False, TE=False, ET=False):
        self.log = log
        self.log.info("Initialising.")

        self.frequencies = frequencies
        self.nmap = len(self.frequencies)
        self.nfreq = len(np.unique(self.frequencies))
        self.nxfreq = self.nfreq*(self.nfreq+1) // 2
        self.nxspec = self.nmap*(self.nmap-1) // 2
        self.xspec2xfreq = self._xspec2xfreq()

        self.TT, self.EE, self.TE, self.ET = TT, EE, TE, ET
        self._is_mode = [TT,EE,TE,ET] #TT,EE,TE,ET
        self.log.debug("mode = {}".format(self._is_mode))
        self.log.debug("frequencies = {}".format(self.frequencies))
        
        # Multipole ranges
        filename = os.path.join(data_folder,multipoles_range_file)
        self.lmins,self.lmaxs = self._set_multipole_ranges( filename)
        self.lmax = np.max([max(l) for l in self.lmaxs])
        
        # Data
        basename = os.path.join(data_folder,xspectra_basename)
        self.dldata = self._read_dl_xspectra(basename, field=1)
        
        # Weights
        basename = os.path.join(data_folder,xerrors_basename)
        dlsig = self._read_dl_xspectra(basename, field=2)
        dlsig[dlsig == 0] = np.inf
        self.dlweight = 1.0 / dlsig ** 2
        
        # Inverted Covariance matrix
        filename = self._get_matrix_name( data_folder, covariance_matrix_basename)
        self.log.info( "Read {}".format( filename))
        self.invkll = self._read_invcovmatrix(filename)

        # Foregrounds
        self.fgs = []  # list of foregrounds per mode [TT,EE,TE,ET]
        # Init foregrounds TT
        fgsTT = []
        if self.TT:
            fgsTT.append(fg.ps_radio(self.lmax, self.frequencies))
            fgsTT.append(fg.ps_dusty(self.lmax, self.frequencies))

            fg_lookup = {
                "dust": fg.dust_model,
                "SZ": fg.sz_model,
                "CIB": fg.cib_model,
                "kSZ": fg.ksz_model,
                "SZxCIB": fg.szxcib_model,
            }
            for name, model in fg_lookup.items():
                if not foregrounds.get(name):
                    continue
                self.log.debug("Adding '{}' foreground for TT".format(name))
                filename = os.path.join(data_folder, foregrounds.get(name))
                kwargs = dict(mode="TT") if name == "dust" else {}
                fgsTT.append(model(self.lmax, self.frequencies, filename, **kwargs))
        self.fgs.append(fgsTT)

        # Get dust filename
        dust_filename = (
            os.path.join(data_folder, foregrounds.get("dust"))
            if foregrounds.get("dust")
            else None
        )

        # Init foregrounds EE
        fgsEE = []
        if self.EE:
            if dust_filename:
                fgsEE.append(fg.dust_model(self.lmax, self.frequencies, dust_filename, mode="EE"))
        self.fgs.append(fgsEE)

        # Init foregrounds TE
        fgsTE = []
        if self.TE:
            if dust_filename:
                fgsTE.append(fg.dust_model(self.lmax, self.frequencies, dust_filename, mode="TE"))
        self.fgs.append(fgsTE)
        fgsET = []
        if self.ET:
            if dust_filename:
                fgsET.append(fg.dust_model(self.lmax, self.frequencies, dust_filename, mode="ET"))
        self.fgs.append(fgsET)

    def _xspec2xfreq( self):
        list_fqs = []
        for f1 in range(self.nfreq):
            for f2 in range(f1, self.nfreq):
                list_fqs.append((f1, f2))

        freqs = list(np.unique(self.frequencies))
        spec2freq = []
        for m1 in range(self.nmap):
            for m2 in range(m1 + 1, self.nmap):
                f1 = freqs.index(self.frequencies[m1])
                f2 = freqs.index(self.frequencies[m2])
                spec2freq.append(list_fqs.index((f1, f2)))
        
        return spec2freq

    def _set_multipole_ranges(self, filename):
        """
        Return the (lmin,lmax) for each cross-spectra for each mode (TT, EE, TE, ET)
        array(nmode,nxspec)
        """
        self.log.debug("Define multipole ranges")
        if not os.path.exists( filename):
            raise ValueError( "File missing {}".format(filename))

        lmins = []
        lmaxs = []
        for hdu in [0, 1, 3, 3]:  # file HDU [TT,EE,BB,TE]
            tags = ["TT","EE","BB","TE","TB","EB"]
            self.log.debug("%s" %(tags[hdu]))
            data = fits.getdata(filename, hdu + 1)
            lmins.append(np.array(data.field(0), int))
            lmaxs.append(np.array(data.field(1), int))
            self.log.debug( "lmin: {}".format(np.array(data.field(0), int)))
            self.log.debug( "lmax: {}".format(np.array(data.field(1), int)))

        return( lmins, lmaxs)

    def _read_dl_xspectra(self, basename, field=1):
        """
        Read xspectra from Xpol [Dl in K^2]
        Output: Dl in muK^2
        """
        self.log.debug("Reading cross-spectra {}".format("errors" if field == 2 else ""))

        dldata = []
        for m1 in range(self.nmap):
            for m2 in range(m1 + 1, self.nmap):
                tmpcl = []
                for mode, hdu in {"TT": 1, "EE": 2, "TE": 4, "ET": 4}.items():
                    filename = "{}_{}_{}.fits".format(basename, m1, m2)
                    if mode == "ET":
                        filename = "{}_{}_{}.fits".format(basename, m2, m1)
                    if not os.path.exists( filename):
                        raise ValueError( "File missing {}".format(filename))
                    data = fits.getdata(filename, hdu)
                    ell = np.array(data.field(0), int)
                    datacl = np.zeros(np.max(ell) + 1)
                    datacl[ell] = data.field(field) * 1e12
                    tmpcl.append(datacl[:self.lmax+1])

                dldata.append(tmpcl)

        return np.transpose(np.array(dldata), (1, 0, 2))

    def _read_invcovmatrix(self,filename):
        """
        Read xspectra inverse covmatrix from Xpol [Dl in K^-4]
        Output: invkll [Dl in muK^-4]
        """
        self.log.debug("Covariance matrix file: {}".format(filename))
        if not os.path.exists( filename):
            raise ValueError( "File missing {}".format(filename))

        data = fits.getdata(filename).field(0)
        nel = int(np.sqrt(len(data)))
        data = data.reshape((nel, nel)) / 1e24  # muK^-4

        nell = self._get_matrix_size()
        if nel != nell:
            raise ValueError( "Incoherent covariance matrix (read:%d, expected:%d)" % (nel, nell))
        
        return data

    def _get_matrix_name( self, data_folder, basename):
        ext = "_"
        for i,e in enumerate( ["TT","EE","TE","ET"]):
            if self._is_mode[i]: ext += e
        return os.path.join(data_folder, basename + ext + ".fits")

    def _get_matrix_size( self):
        nell = 0
        
        #TT,EE,TEET
        for m in range(3):
            if self._is_mode[m]:
                nells = self.lmaxs[m] - self.lmins[m] + 1
                nell += np.sum([nells[self.xspec2xfreq.index(k)] for k in range(self.nxfreq)])
        
        return nell
    
    def _select_spectra(self, cl, mode=0):
        """
        Cut spectra given Multipole Ranges and flatten
        Return: list
        """
        acl = np.asarray(cl)
        xl = []
        for xf in range(self.nxfreq):
            lmin = self.lmins[mode][self.xspec2xfreq.index(xf)]
            lmax = self.lmaxs[mode][self.xspec2xfreq.index(xf)]
            xl += list(acl[xf, lmin : lmax + 1])
        return xl
    
    def _xspectra_to_xfreq(self, cl, weight, normed=True):
        """
        Average cross-spectra per cross-frequency
        """
        xcl = np.zeros((self.nxfreq, self.lmax + 1))
        xw8 = np.zeros((self.nxfreq, self.lmax + 1))
        for xs in range(self.nxspec):
            xcl[self.xspec2xfreq[xs]] += weight[xs] * cl[xs]
            xw8[self.xspec2xfreq[xs]] += weight[xs]

        xw8[xw8 == 0] = np.inf
        if normed:
            return xcl / xw8
        else:
            return (xcl, xw8)
    
    def _compute_residuals(self, pars, dlth, mode=0):
        
        # nuisances
        cal = []
        for m1 in range(self.nmap):
            for m2 in range(m1 + 1, self.nmap):
                cal.append(pars["Aplanck"] ** 2 * (1.0 + pars["c%d" % m1] + pars["c%d" % m2]))
        
        # Data
        dldata = self.dldata[mode]
        
        # Model
        dlmodel = [dlth[mode]] * self.nxspec
        for fg in self.fgs[mode]:
            dlmodel += fg.compute_dl(pars)
        
        # Compute Rl = Dl - Dlth
        Rspec = np.array([dldata[xs] - cal[xs] * dlmodel[xs] for xs in range(self.nxspec)])
        
        return Rspec
    
    def compute_chi2( self, cl, **params_values):
        """
        Compute likelihood from model out of Boltzmann code
        Units: Dl in muK^2

        Parameters
        ----------
        pars: dict
              parameter values
        cl: array or arr2d
              CMB power spectrum (Cl in muK^2)

        Returns
        -------
        lnL: float
            Log likelihood for the given parameters -2ln(L)
        """
        
        # cl_boltz from Boltzmann (Cl in muK^2)
        lth = np.arange(self.lmax + 1)
        clth = np.asarray(cl)
        dlth = np.asarray(clth[:,lth]*lth*(lth+1)/2./np.pi)
        
        # Create Data Vector
        Xl = []
        if self.TT:
            # compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals(params_values, dlth, mode=0)
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self.dlweight[0])
            # select multipole range
            Xl += self._select_spectra(Rl, mode=0)
        
        if self.EE:
            # compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals(params_values, dlth, mode=1)
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self.dlweight[1])
            # select multipole range
            Xl += self._select_spectra(Rl, mode=1)
        
        if self.TE or self.ET:
            Rl = 0
            Wl = 0
            # compute residuals Rl = Dl - Dlth
            if self.TE:
                Rspec = self._compute_residuals(params_values, dlth, mode=2)
                RlTE, WlTE = self._xspectra_to_xfreq(Rspec, self.dlweight[2], normed=False)
                Rl = Rl + RlTE
                Wl = Wl + WlTE
            if self.ET:
                Rspec = self._compute_residuals(params_values, dlth, mode=3)
                RlET, WlET = self._xspectra_to_xfreq(Rspec, self.dlweight[3], normed=False)
                Rl = Rl + RlET
                Wl = Wl + WlET
            # select multipole range
            Xl += self._select_spectra(Rl / Wl, mode=2)
        
        Xl = np.array(Xl)
        chi2 = Xl.dot(self.invkll).dot(Xl)
        
        return chi2

# ------------------------------------------------------------------------------------------------





# ------------------------------------------------------------------------------------------------
# External tools
# ------------------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------------------







#------------------------------------------------------------------------------------------------
#Binning
#------------------------------------------------------------------------------------------------
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
        Average spectra with defined bins
        can be weighted by `l(l+1)/2pi`.
        Return Cb
        """
        spectra = np.asarray(spectra)
        minlmax = min([spectra.shape[-1] - 1,self.lmax])
        
        _p, _q = self._bin_operators(Dl=Dl)
        return np.dot(spectra[..., :minlmax+1], _p.T[:minlmax+1,...])
    
    def bin_covariance( self, clcov):
        """
        Average covariance with defined bins
        """
        p,q = self._bin_operators(cov=True)
        return( np.matmul( p, np.matmul( clcov, q)))
