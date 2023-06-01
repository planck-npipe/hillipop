# FOREGROUNDS V4 (Planck 2020)
import astropy.io.fits as fits
import os
import numpy as np
import itertools
from cobaya.log import HasLogger, LoggedError

t_cmb = 2.72548
k_b = 1.3806503e-23
h_pl = 6.626068e-34

# ------------------------------------------------------------------------------------------------
# Foreground class
# ------------------------------------------------------------------------------------------------
class fgmodel(HasLogger):
    """
    Class of foreground model for the Hillipop likelihood
    Units: Dl in muK^2
    Should return the model in Dl for a foreground emission given the parameters for all correlation of frequencies
    """

    # Planck values
    feff = 143

    # MJy.sr-1 -> Kcmb for (100,143,217) GHz
    gnu = {100: 244.059, 143: 371.658, 217: 483.485}

    fsz = {100:100.24, 143: 143, 217: 222}
    fdust = {100:105.25, 143:148.23, 217:229.1, 353:372.19} #alpha=4 from [Planck 2013 IX]
    fcib = fdust
    fsyn = {100:100,143:143,217:217}

    def _f_tsz( self, freq):
        # Freq in GHz
        nu = freq*1e9
        xx=h_pl*nu/(k_b*t_cmb)
        return xx*( 1/np.tanh(xx/2.) ) - 4

    def _f_Planck( self, f, T):
        # Freq in GHz
        nu = f*1e9
        xx  = h_pl*nu /(k_b*T)
        return (nu**3.)/(np.exp(xx)-1.)

    def _dBdT(self, f):
        # Freq in GHz
        nu  = f*1e9
        xx  = h_pl*nu /(k_b*t_cmb)
        return (nu)**4 * np.exp(xx) / (np.exp(xx)-1.)**2.

    def _tszRatio( self, f, f0):
        return self._f_tsz(f)/self._f_tsz(f0)

    def _cibRatio( self, f, f0, beta, T=9.7):
        return (f/f0)**beta * (self._f_Planck(f,T)/self._f_Planck(f0,T)) / ( self._dBdT(f)/self._dBdT(f0) )

    def _dustRatio( self, f, f0, beta=1.5, T=19.6):
        return (f/f0)**beta * (self._f_Planck(f,T)/self._f_Planck(f0,T)) / ( self._dBdT(f)/self._dBdT(f0) )

    def __init__(self, lmax, freqs, mode="TT", auto=False, **kwargs):
        """
        Create model for foreground
        """
        self.mode = mode
        self.lmax = lmax
        self.freqs = freqs
        self.name = None

        ell = np.arange(lmax + 1)
        self.ll2pi = ell * (ell + 1) / (3000*3001)

        # Build the list of cross frequencies
        self._cross_frequencies = list(
            itertools.combinations_with_replacement(freqs, 2)
            if auto
            else itertools.combinations(freqs, 2)
        )
        self.set_logger()
        pass

    def _gen_dl_powerlaw( self, alpha, lnorm=3000):
        """
        Generate power-law Dl template
        Input: alpha in Cl
        """
        lmax = self.lmax if lnorm is None else max(self.lmax,lnorm)
        ell = np.arange( 2, lmax+1)

        template = np.zeros( lmax+1)
        template[np.array(ell,int)] = ell*(ell+1)/2/np.pi * ell**(alpha)

        #normalize l=3000
        if lnorm is not None:
            template = template / template[lnorm]

        return template[:self.lmax+1]

    def _read_dl_template( self, filename, lnorm=3000):
        """
        Read FG template (in Dl, muK^2)
        WARNING: need to check file before reading...
        """

        if not os.path.exists(filename):
            raise LoggedError( self.log, "Missing file: %s", self.filename)

        #read dl template
        l,data = np.loadtxt( filename, unpack=True)
        l = np.array(l,int)
        self.log.debug( "Template: {}".format(filename))

        if max(l) < self.lmax:
            self.log.info( "WARNING: template {} has lower lmax (filled with 0)".format(filename))
        template = np.zeros( max(self.lmax,max(l)) + 1)
        template[l] = data

        #normalize l=3000
        if lnorm is not None:
            template = template / template[lnorm]
        
        return template[:self.lmax+1]

    def compute_dl(self, pars):
        """
        Return spectra model for each cross-spectra
        """
        pass


# ------------------------------------------------------------------------------------------------



# Point Sources
class ps(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "PS"

    def compute_dl(self, pars):
        dl_ps = []
        for f1, f2 in self._cross_frequencies:
            dl_ps.append( pars["Aps_{}x{}".format(f1,f2)] * self.ll2pi)

        if self.mode == "TT":
            return np.array(dl_ps)
        else:
            return 0.



# Radio Point Sources (v**alpha)
class ps_radio(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "PS radio"

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(
                self.ll2pi
                * (self.fsyn[f1]*self.fsyn[f2]/self.feff**2)**pars['alpha_s']
                / ( self._dBdT(f1)*self._dBdT(f2)/self._dBdT(self.feff)**2 )
            )

        if self.mode == "TT":
            return pars["Aradio"] * np.array(dl)
        else:
            return 0.


# Infrared Point Sources
class ps_dusty(fgmodel):
    def __init__(self, lmax, freqs, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "PS dusty"

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append(
                self.ll2pi
                * self._cibRatio(self.fcib[f1],self.feff,pars['beta_dusty'])
                * self._cibRatio(self.fcib[f2],self.feff,pars['beta_dusty'])
            )

        if self.mode == "TT":
            return pars["Adusty"] * np.array(dl)
        else:
            return 0.


# Galactic Dust
class dust(fgmodel):
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "Dust"
        
        self.dlg = []
        if filename is None:
            alpha_dust = -2.4 if mode == "EE" else -2.5
            dlg = self._gen_dl_powerlaw( alpha_dust, lnorm=80)

            #amplitude for Cl dust at 353GHz and l=80 on XXL for fixed alpha
            self.A353 = {"TT": {100:39000,143:19000,217:11500},
                         "EE": {100:463,143:300,217:160},
                         "TE": {100:1150,143:850,217:450},
                         "ET": {100:1150,143:850,217:450}}

            for f1, f2 in self._cross_frequencies:
                self.dlg.append(dlg*self.A353[self.mode][np.max([f1,f2])])
        else:
            hdr = ["ell","100x100","100x143","100x217","143x143","143x217","217x217"]
            data = np.loadtxt( f"{filename}_{mode}.txt").T
            l = np.array(data[0],int)
            for f1, f2 in self._cross_frequencies:
                tmpl = np.zeros(max(l) + 1)
                tmpl[l] = data[hdr.index(f"{f1}x{f2}")]
                self.dlg.append( tmpl[:lmax+1])

        self.dlg = np.array(self.dlg)

    def compute_dl(self, pars):
        if self.mode == "TT":
            A = B = {100:pars["Ad100T"],143:pars["Ad143T"],217:pars["Ad217T"]}
        if self.mode == "EE":
            A = B = {100:pars["Ad100P"],143:pars["Ad143P"],217:pars["Ad217P"]}
        if self.mode == "TE":
            A = {100:pars["Ad100T"],143:pars["Ad143T"],217:pars["Ad217T"]}
            B = {100:pars["Ad100P"],143:pars["Ad143P"],217:pars["Ad217P"]}
        if self.mode == "ET":
            A = {100:pars["Ad100P"],143:pars["Ad143P"],217:pars["Ad217P"]}
            B = {100:pars["Ad100T"],143:pars["Ad143T"],217:pars["Ad217T"]}

        Ad = [A[f1]*B[f2] for f1, f2 in self._cross_frequencies]

        return np.array(Ad)[:, None] * self.dlg


class dust_model(fgmodel):
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "Dust model"
        
        self.dlg = []
        hdr = ["ell","100x100","100x143","100x217","143x143","143x217","217x217"]
        data = np.loadtxt( f"{filename}_{mode}.txt").T
        l = np.array(data[0],int)
        for f1, f2 in self._cross_frequencies:
            tmpl = np.zeros(max(l) + 1)
            tmpl[l] = data[hdr.index(f"{f1}x{f2}")]
            self.dlg.append( tmpl[:lmax+1])
        self.dlg = np.array(self.dlg)

    def compute_dl(self, pars):
        if   self.mode == "TT": Adust = pars['AdustT']*pars['AdustT']
        elif self.mode == "TE": Adust = pars['AdustT']*pars['AdustP']
        elif self.mode == "ET": Adust = pars['AdustP']*pars['AdustT']
        elif self.mode == "EE": Adust = pars['AdustP']*pars['AdustP']
        else: Adust = 0.

        dl = []
        for xf, (f1, f2) in enumerate(self._cross_frequencies):
            dl.append( Adust * self.dlg[xf]
                       * self._dustRatio(self.fdust[f1],353,pars['beta_dust'],19.6)
                       * self._dustRatio(self.fdust[f2],353,pars['beta_dust'],19.6)
                       )
        return np.array(dl)



# CIB model (one spectrum per xfreq)
class cib_model(fgmodel):
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "clustered CIB"

        #check effective freqs
        for f in freqs:
            if f not in self.fcib:
                raise ValueError( f"Missing CIB effective frequency for {f}")

        if filename is None:
            alpha_cib = -1.3
            self.dl_cib = self._gen_dl_powerlaw( alpha_cib)
        else:
            self.dl_cib = self._read_dl_template( filename)

    def compute_dl(self, pars):
        dl = []
        for f1, f2 in self._cross_frequencies:
            dl.append( self.dl_cib
                       * self._cibRatio(self.fcib[f1],self.feff,pars['beta_cib'])
                       * self._cibRatio(self.fcib[f2],self.feff,pars['beta_cib'])
                       )
        if self.mode == "TT":
            return pars["Acib"] * np.array(dl)
        else:
            return 0.

# tSZ (one spectrum for all freqs)
class tsz_model(fgmodel):
    def __init__(self, lmax, freqs, filename="", mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        # template: Dl=l(l+1)/2pi Cl, units uK at 143GHz
        self.name = "tSZ"

        #check effective freqs for SZ
        for f in freqs:
            if f not in self.fsz:
                raise ValueError( f"Missing SZ effective frequency for {f}")

        # read Dl template (normalized at l=3000)
        sztmpl = self._read_dl_template(filename)

        self.dl_sz = []
        for f1, f2 in self._cross_frequencies:
            self.dl_sz.append( sztmpl[: lmax + 1]
                               * self._tszRatio(self.fsz[f1],self.feff)
                               * self._tszRatio(self.fsz[f2],self.feff)
                               )
        self.dl_sz = np.array(self.dl_sz)

    def compute_dl(self, pars):
        return pars["Atsz"] * self.dl_sz


# kSZ (one spectrum for all freqs)
class ksz_model(fgmodel):
    def __init__(self, lmax, freqs, filename="", mode="TT", auto=False):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        # template: Dl=l(l+1)/2pi Cl, units uK
        self.name = "kSZ"

        # read Dl template (normalized at l=3000)
        ksztmpl = self._read_dl_template(filename)

        self.dl_ksz = []
        for f1, f2 in self._cross_frequencies:
            self.dl_ksz.append(ksztmpl[: lmax + 1])
        self.dl_ksz = np.array(self.dl_ksz)

    def compute_dl(self, pars):
        if self.mode == "TT":
            return pars["Aksz"] * self.dl_ksz
        else:
            return 0.


# SZxCIB model (one spectrum per xfreq)
class szxcib_model(fgmodel):
    def __init__(self, lmax, freqs, filename=None, mode="TT", auto=False, **kwargs):
        super().__init__(lmax, freqs, mode=mode, auto=auto)
        self.name = "SZxCIB"

        #check effective freqs for SZ
        for f in freqs:
            if f not in self.fsz:
                raise ValueError( f"Missing SZ effective frequency for {f}")

        #check effective freqs for dust
        for f in freqs:
            if f not in self.fcib:
                raise ValueError( f"Missing Dust effective frequency for {f}")

        self._is_template = filename
        if self._is_template:
            self.x_tmpl = self._read_dl_template(filename)
        elif "filenames" in kwargs:
            self.x_tmpl = self._read_dl_template(kwargs["filenames"][0])*self._read_dl_template(kwargs["filenames"][1])
        else:
            raise ValueError( f"Missing template for SZxCIB")
            
    def compute_dl(self, pars):
        dl_szxcib = []
        for f1, f2 in self._cross_frequencies:
            dl_szxcib.append( self.x_tmpl * np.sqrt(pars["Acib"]*pars["Atsz"]) * (
                self._tszRatio(self.fsz[f2],self.feff) * self._cibRatio(self.fcib[f1], self.feff, pars['beta_cib']) +
                self._tszRatio(self.fsz[f1],self.feff) * self._cibRatio(self.fcib[f2], self.feff, pars['beta_cib'])
                )
            )

        if self.mode == "TT":
            return -1. * pars["xi"] * np.array(dl_szxcib)
        else:
            return 0.
