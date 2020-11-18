#
# HILLIPOP
#
# Sep 2020   - M. Tristram -
import os
from typing import Optional

import astropy.io.fits as fits
import numpy as np
from cobaya.conventions import _packages_path
from cobaya.likelihoods._base_classes import _InstallableLikelihood
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists

from hillipop import foregrounds_v3 as fg
from hillipop import tools








# ------------------------------------------------------------------------------------------------
# Likelihood
# ------------------------------------------------------------------------------------------------

class _hlp_likelihood(_InstallableLikelihood):
    """High-L Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency
    maps

    """

    multipoles_range_file: Optional[str]
    frequencies: Optional[list]
    foregrounds: list

    _is_mode = {"TT":True,"EE":True,"TE":True,"ET":True}

    def initialize(self):
        self.log.info("Initialising.")
        
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, _packages_path, None)):
            raise LoggedError(
                self.log,
                "No path given to Hillipop data. Set the likelihood property 'path' or the common property '%s'.",
                _packages_path,
            )
        
        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, "data")
        )
        
        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. Check the given path [%s].",
                self.data_folder,
            )

        self.nmap = len(self.frequencies)
        self.nfreq = len(np.unique(self.frequencies))
        self.nxfreq = self.nfreq*(self.nfreq+1) // 2
        self.nxspec = self.nmap*(self.nmap-1) // 2
        self.xspec2xfreq = self._xspec2xfreq()

        self.log.debug("mode = {}".format(self._is_mode))
        self.log.debug("frequencies = {}".format(self.frequencies))
        
        # Multipole ranges
        filename = os.path.join(self.data_folder,self.multipoles_range_file)
        self.lmins,self.lmaxs = self._set_multipole_ranges( filename)
        self.lmax = np.max([max(l) for l in self.lmaxs])
        
        # Data
        basename = os.path.join(self.data_folder,self.xspectra_basename)
        self.dldata = self._read_dl_xspectra(basename, field=1)
        
        # Weights
        basename = os.path.join(self.data_folder,self.xspectra_errors_basename)
        dlsig = self._read_dl_xspectra(basename, field=2)
        dlsig[dlsig == 0] = np.inf
        self.dlweight = 1.0 / dlsig ** 2
        
        # Inverted Covariance matrix
        filename = self._get_matrix_name( self.data_folder, self.covariance_matrix_basename)
        self.log.info( "Read {}".format( filename))
        self.invkll = self._read_invcovmatrix(filename)

        # Foregrounds
        self.fgs = []  # list of foregrounds per mode [TT,EE,TE,ET]
        # Init foregrounds TT
        fgsTT = []
        if self._is_mode["TT"]:
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
                if not self.foregrounds.get(name):
                    continue
                self.log.debug("Adding '{}' foreground for TT".format(name))
                filename = os.path.join(self.data_folder, self.foregrounds.get(name))
                kwargs = dict(mode="TT") if name == "dust" else {}
                fgsTT.append(model(self.lmax, self.frequencies, filename, **kwargs))
        self.fgs.append(fgsTT)

        # Get dust filename
        dust_filename = (
            os.path.join(self.data_folder, self.foregrounds.get("dust"))
            if self.foregrounds.get("dust")
            else None
        )

        # Init foregrounds EE
        fgsEE = []
        if self._is_mode["EE"]:
            if dust_filename:
                fgsEE.append(fg.dust_model(self.lmax, self.frequencies, dust_filename, mode="EE"))
        self.fgs.append(fgsEE)

        # Init foregrounds TE
        fgsTE = []
        if self._is_mode["TE"]:
            if dust_filename:
                fgsTE.append(fg.dust_model(self.lmax, self.frequencies, dust_filename, mode="TE"))
        self.fgs.append(fgsTE)
        fgsET = []
        if self._is_mode["ET"]:
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
        for e in ["TT","EE","TE","ET"]:
            if self._is_mode[e]: ext += e
        return os.path.join(data_folder, basename + ext + ".fits")

    def _get_matrix_size( self):
        nell = 0
        
        #TT,EE,TEET
        for im,m in enumerate(["TT","EE","TE"]):
            if self._is_mode[m]:
                nells = self.lmaxs[im] - self.lmins[im] + 1
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
        clth = np.asarray(cl)[[0,1,3,3]]  #select TT,EE,TE,TE
        dlth = np.asarray(clth[:,lth]*lth*(lth+1)/2./np.pi)
        
        # Create Data Vector
        Xl = []
        if self._is_mode["TT"]:
            # compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals(params_values, dlth, mode=0)
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self.dlweight[0])
            # select multipole range
            Xl += self._select_spectra(Rl, mode=0)
        
        if self._is_mode["EE"]:
            # compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals(params_values, dlth, mode=1)
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self.dlweight[1])
            # select multipole range
            Xl += self._select_spectra(Rl, mode=1)
        
        if self._is_mode["TE"] or self._is_mode["ET"]:
            Rl = 0
            Wl = 0
            # compute residuals Rl = Dl - Dlth
            if self._is_mode["TE"]:
                Rspec = self._compute_residuals(params_values, dlth, mode=2)
                RlTE, WlTE = self._xspectra_to_xfreq(Rspec, self.dlweight[2], normed=False)
                Rl = Rl + RlTE
                Wl = Wl + WlTE
            if self._is_mode["ET"]:
                Rspec = self._compute_residuals(params_values, dlth, mode=3)
                RlET, WlET = self._xspectra_to_xfreq(Rspec, self.dlweight[3], normed=False)
                Rl = Rl + RlET
                Wl = Wl + WlET
            # select multipole range
            Xl += self._select_spectra(Rl / Wl, mode=2)
        
        Xl = np.array(Xl)
        chi2 = Xl.dot(self.invkll).dot(Xl)

        self.log.debug( "chi2=%f / nodf=%d" % (chi2, len(Xl)))
        return chi2


    def get_requirements(self):
        return dict(Cl={mode: self.lmax for mode in ["tt","ee","te"]})

    def logp(self, **params_values):
        cl = self.theory.get_Cl(ell_factor=False)
        return self.loglike(cl, **params_values)

    def loglike(self, cl, **params_values):
        """
        Compute likelihood from model out of Boltzmann code
        Units: Dl in muK^2

        Parameters
        ----------
        pars: dict
              parameter values
        cl: dict
              CMB power spectrum (Cl in µK^2)

        Returns
        -------
        lnL: float
            Log likelihood for the given parameters -2ln(L)
        """

        # cl_boltz from Boltzmann (Cl in muK^2)
        lth = np.arange(self.lmax + 1)
        clth = np.zeros( (4,self.lmax+1) )
        clth[0] = cl["tt"][lth]
        clth[1] = cl["ee"][lth]
        clth[3] = cl["te"][lth]

        chi2 = self.compute_chi2( clth, **params_values)

        return -0.5 * chi2

# ------------------------------------------------------------------------------------------------





class TTTEEE(_hlp_likelihood):
    """High-L Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency
    maps

    """

    _is_mode = {"TT":True,"EE":True,"TE":True,"ET":True}


class TT(_hlp_likelihood):
    """High-L Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency
    maps

    """

    _is_mode = {"TT":True,"EE":False,"TE":False,"ET":False}



class EE(_hlp_likelihood):
    """High-L Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency
    maps

    """

    _is_mode = {"TT":False,"EE":True,"TE":False,"ET":False}


class TE(_hlp_likelihood):
    """High-L Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency
    maps

    """

    _is_mode = {"TT":False,"EE":False,"TE":True,"ET":True}


