#
# HILLIPOP
#
# Sep 2020   - M. Tristram -
import glob
import logging
import os
import re
from itertools import combinations
from typing import Optional

import astropy.io.fits as fits
import numpy as np
from cobaya.conventions import data_path, packages_path_input
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError

from . import foregrounds as fg
from . import tools

# list of available foreground models
fg_list = {
    "sbpx": fg.subpix,
    "ps": fg.ps,
    "dust": fg.dust,
    "dust_model": fg.dust_model,
    "sync": fg.sync_model,
    "ksz": fg.ksz_model,
    "ps_radio": fg.ps_radio,
    "ps_dusty": fg.ps_dusty,
    "cib": fg.cib_model,
    "tsz": fg.tsz_model,
    "szxcib": fg.szxcib_model,
}


# ------------------------------------------------------------------------------------------------
# Likelihood
# ------------------------------------------------------------------------------------------------

data_url = "https://portal.nersc.gov/cfs/cmb/planck2020/likelihoods"


class _HillipopLikelihood(InstallableLikelihood):
    multipoles_range_file: Optional[str]
    xspectra_basename: Optional[str]
    covariance_matrix_file: Optional[str]
    foregrounds: Optional[list]

    def initialize(self):
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, packages_path_input, None)):
            raise LoggedError(
                self.log,
                "No path given to Hillipop data. Set the likelihood property "
                f"'path' or the common property '{packages_path_input}'.",
            )

        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, data_path)
        )

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. Check the given path [%s].",
                self.data_folder,
            )

        self.frequencies = [100, 100, 143, 143, 217, 217]
        self._mapnames = ["100A", "100B", "143A", "143B", "217A", "217B"]
        self._nmap = len(self.frequencies)
        self._nfreq = len(np.unique(self.frequencies))
        self._nxfreq = self._nfreq * (self._nfreq + 1) // 2
        self._nxspec = self._nmap * (self._nmap - 1) // 2
        self._xspec2xfreq = self._xspec2xfreq()
        self.log.debug(f"frequencies = {self.frequencies}")

        # Get likelihood name and add the associated mode
        likelihood_name = self.__class__.__name__
        likelihood_modes = [likelihood_name[i : i + 2] for i in range(0, len(likelihood_name), 2)]
        self._is_mode = {mode: mode in likelihood_modes for mode in ["TT", "TE", "EE", "BB"]}
        self._is_mode["ET"] = self._is_mode["TE"]
        self.log.debug(f"mode = {self._is_mode}")

        # Multipole ranges
        filename = os.path.join(self.data_folder, self.multipoles_range_file)
        self._lmins, self._lmaxs = self._set_multipole_ranges(filename)
        self.lmax = np.max([max(l) for l in self._lmaxs.values()])

        # Data
        basename = os.path.join(self.data_folder, self.xspectra_basename)
        self._dldata = self._read_dl_xspectra(basename)

        # Weights
        dlsig = self._read_dl_xspectra(basename, hdu=2)
        for m,w8 in dlsig.items(): w8[w8==0] = np.inf
        self._dlweight = {k:1/v**2 for k,v in dlsig.items()}
#        self._dlweight = np.ones(np.shape(self._dldata))

        # Inverted Covariance matrix
        filename = os.path.join(self.data_folder, self.covariance_matrix_file)
        # Sanity check
        m = re.search(".*_(.+?).fits", self.covariance_matrix_file)
        if not m or likelihood_name != m.group(1):
            raise LoggedError(
                self.log,
                "The covariance matrix mode differs from the likelihood mode. "
                f"Check the given path [{self.covariance_matrix_file}]",
            )
        self._invkll = self._read_invcovmatrix(filename)
        self._invkll = self._invkll.astype('float32')

        # Foregrounds
        self.fgs = {}  # list of foregrounds per mode [TT,EE,TE,ET]
        # Init foregrounds TT
        fgsTT = []
        if self._is_mode["TT"]:
            for name in self.foregrounds["TT"].keys():
                if name not in fg_list.keys():
                    raise LoggedError(self.log, f"Unkown foreground model '{name}'!")
                self.log.debug(f"Adding '{name}' foreground for TT")
                kwargs = dict(lmax=self.lmax, freqs=self.frequencies, mode="TT")
                if isinstance(self.foregrounds["TT"][name], str):
                    kwargs["filename"] = os.path.join(self.data_folder, self.foregrounds["TT"][name])
                elif name == "szxcib":
                    filename_tsz = self.foregrounds["TT"]["tsz"] and os.path.join(self.data_folder, self.foregrounds["TT"]["tsz"])
                    filename_cib = self.foregrounds["TT"]["cib"] and os.path.join(self.data_folder, self.foregrounds["TT"]["cib"])
                    kwargs["filenames"] = (filename_tsz,filename_cib)
                fgsTT.append(fg_list[name](**kwargs))
        self.fgs['TT'] = fgsTT

        # Init foregrounds EE
        fgsEE = []
        if self._is_mode["EE"]:
            for name in self.foregrounds["EE"].keys():
                if name not in fg_list.keys():
                    raise LoggedError(self.log, f"Unkown foreground model '{name}'!")
                self.log.debug(f"Adding '{name}' foreground for EE")
                kwargs = dict(lmax=self.lmax, freqs=self.frequencies)
                if isinstance(self.foregrounds["EE"][name], str):
                    kwargs["filename"] = os.path.join(self.data_folder, self.foregrounds["EE"][name])
                fgsEE.append(fg_list[name](mode="EE", **kwargs))
        self.fgs['EE'] = fgsEE

        # Init foregrounds TE
        fgsTE = []
        fgsET = []
        if self._is_mode["TE"]:
            for name in self.foregrounds["TE"].keys():
                if name not in fg_list.keys():
                    raise LoggedError(self.log, f"Unkown foreground model '{name}'!")
                self.log.debug(f"Adding '{name}' foreground for TE")
                kwargs = dict(lmax=self.lmax, freqs=self.frequencies)
                if isinstance(self.foregrounds["TE"][name], str):
                    kwargs["filename"] = os.path.join(self.data_folder, self.foregrounds["TE"][name])
                fgsTE.append(fg_list[name](mode="TE", **kwargs))
                fgsET.append(fg_list[name](mode="ET", **kwargs))
        self.fgs['TE'] = fgsTE
        self.fgs['ET'] = fgsET

        self.log.info("Initialized!")

    def _xspec2xfreq(self):
        list_fqs = []
        for f1 in range(self._nfreq):
            for f2 in range(f1, self._nfreq):
                list_fqs.append((f1, f2))

        freqs = list(np.unique(self.frequencies))
        spec2freq = []
        for m1 in range(self._nmap):
            for m2 in range(m1 + 1, self._nmap):
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
        if not os.path.exists(filename):
            raise ValueError(f"File missing {filename}")

        tags = ["TT", "EE", "BB", "TE"] 
        lmins = {}
        lmaxs = {}
        with fits.open( filename) as hdus:
            for hdu in hdus[1:]:
                tag = hdu.header['spec']
                lmins[tag] = hdu.data.LMIN
                lmaxs[tag] = hdu.data.LMAX
                if self._is_mode[tag]:
                    self.log.debug(f"{tag}")
                    self.log.debug(f"lmin: {lmins[tag]}")
                    self.log.debug(f"lmax: {lmaxs[tag]}")
        lmins["ET"] = lmins["TE"]
        lmaxs["ET"] = lmaxs["TE"]

        return lmins, lmaxs

    def _read_dl_xspectra(self, basename, hdu=1):
        """
        Read xspectra from Xpol [Dl in K^2]
        Output: Dl (TT,EE,TE,ET) in muK^2
        """
        self.log.debug("Reading cross-spectra")
        
        with fits.open(f"{basename}_{self._mapnames[0]}x{self._mapnames[1]}.fits") as hdus:
            nhdu = len( hdus)
        if nhdu < hdu:
            #no sig in file, uniform weight
            self.log.info( "Warning: uniform weighting for combining spectra !")
            dldata = np.ones( (self._nxspec, 4, self.lmax+1))
        else:
            if nhdu == 1: hdu=0 #compatibility
            dldata = []
            for m1, m2 in combinations(self._mapnames, 2):
                data = fits.getdata( f"{basename}_{m1}x{m2}.fits", hdu)*1e12
                tmpcl = list(data[[0,1,3],:self.lmax+1])
                data = fits.getdata( f"{basename}_{m2}x{m1}.fits", hdu)*1e12
                tmpcl.append( data[3,:self.lmax+1])
                dldata.append( tmpcl)

        dldata = np.transpose(np.array(dldata), (1, 0, 2))
        return dict(zip(['TT','EE','TE','ET'],dldata))

    def _read_invcovmatrix(self, filename):
        """
        Read xspectra inverse covmatrix from Xpol [Dl in K^-4]
        Output: invkll [Dl in muK^-4]
        """
        self.log.debug(f"Covariance matrix file: {filename}")
        if not os.path.exists(filename):
            raise ValueError(f"File missing {filename}")

        data = fits.getdata(filename)
        nel = int(np.sqrt(data.size))
        data = data.reshape((nel, nel)) / 1e24  # muK^-4

        nell = self._get_matrix_size()
        if nel != nell:
            raise ValueError(f"Incoherent covariance matrix (read:{nel}, expected:{nell})")

        return data

    def _get_matrix_size(self):
        """
        Compute covariance matrix size given activated mode
        Return: number of multipole
        """
        nell = 0

        # TT,EE,TEET
        for m in ["TT", "EE", "TE"]:
            if self._is_mode[m]:
                nells = self._lmaxs[m] - self._lmins[m] + 1
                nell += np.sum([nells[self._xspec2xfreq.index(k)] for k in range(self._nxfreq)])

        return nell

    def _select_spectra(self, cl, mode):
        """
        Cut spectra given Multipole Ranges and flatten
        Return: list
        """
        acl = np.asarray(cl)
        xl = []
        for xf in range(self._nxfreq):
            lmin = self._lmins[mode][self._xspec2xfreq.index(xf)]
            lmax = self._lmaxs[mode][self._xspec2xfreq.index(xf)]
            xl += list(acl[xf, lmin : lmax + 1])
        return xl

    def _xspectra_to_xfreq(self, cl, weight, normed=True):
        """
        Average cross-spectra per cross-frequency
        """
        xcl = np.zeros((self._nxfreq, self.lmax + 1))
        xw8 = np.zeros((self._nxfreq, self.lmax + 1))
        for xs in range(self._nxspec):
            xcl[self._xspec2xfreq[xs]] += weight[xs] * cl[xs]
            xw8[self._xspec2xfreq[xs]] += weight[xs]

        xw8[xw8 == 0] = np.inf
        if normed:
            return xcl / xw8
        else:
            return xcl, xw8

    def _compute_residuals(self, pars, dlth, mode):
        # Nuisances
        cal = []
        for m1, m2 in combinations(self._mapnames, 2):
            if mode == "TT":
                cal1, cal2 = pars[f"cal{m1}"], pars[f"cal{m2}"]
            elif mode == "EE":
                cal1, cal2 = pars[f"cal{m1}"]*pars[f"pe{m1}"], pars[f"cal{m2}"]*pars[f"pe{m2}"]
            elif mode == "TE":
                cal1, cal2 = pars[f"cal{m1}"], pars[f"cal{m2}"]*pars[f"pe{m2}"]
            elif mode == "ET":
                cal1, cal2 = pars[f"cal{m1}"]*pars[f"pe{m1}"], pars[f"cal{m2}"]
            cal.append(cal1 * cal2 / pars["A_planck"] ** 2)
        
        # Data
        dldata = self._dldata[mode]
        
        # Model
        dlmodel = [dlth[mode]] * self._nxspec
        for fg in self.fgs[mode]:
            dlmodel += fg.compute_dl(pars)
        
        # Compute Rl = Dl - Dlth
        Rspec = np.array([dldata[xs] - cal[xs] * dlmodel[xs] for xs in range(self._nxspec)])

        return Rspec

    def dof( self):
        return len( self._invkll)

    def reduction_matrix(self, mode):
        """
        Reduction matrix

        each column is equal to 1 in the 15 elements corresponding to a cross-power spectrum
        measurement in that multipole and zero elsewhere

        """
        X = np.zeros( (len(self.delta_cl),self.lmax+1) )
        x0 = 0
        for xf in range(self._nxfreq):
            lmin = self._lmins[mode][self._xspec2xfreq.index(xf)]
            lmax = self._lmaxs[mode][self._xspec2xfreq.index(xf)]
            for il,l in enumerate(range(lmin,lmax+1)):
                X[x0+il,l] = 1
            x0 += (lmax-lmin+1)
        
        return X

    def compute_chi2(self, dlth, **params_values):
        """
        Compute likelihood from model out of Boltzmann code
        Units: Dl in muK^2

        Parameters
        ----------
        pars: dict
              parameter values
        dl: array or arr2d
              CMB power spectrum (Dl in muK^2)

        Returns
        -------
        lnL: float
            Log likelihood for the given parameters -2ln(L)
        """

        # cl_boltz from Boltzmann (Cl in muK^2)
#        lth = np.arange(self.lmax + 1)
#        dlth = np.asarray(dl)[:, lth][[0, 1, 3, 3]]  # select TT,EE,TE,TE

        # Create Data Vector
        Xl = []
        if self._is_mode["TT"]:
            # compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals(params_values, dlth, "TT")
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self._dlweight["TT"])
            # select multipole range
            Xl += self._select_spectra(Rl, 'TT')

        if self._is_mode["EE"]:
            # compute residuals Rl = Dl - Dlth
            Rspec = self._compute_residuals(params_values, dlth, "EE")
            # average to cross-spectra
            Rl = self._xspectra_to_xfreq(Rspec, self._dlweight["EE"])
            # select multipole range
            Xl += self._select_spectra(Rl, 'EE')

        if self._is_mode["TE"] or self._is_mode["ET"]:
            Rl = 0
            Wl = 0
            # compute residuals Rl = Dl - Dlth
            if self._is_mode["TE"]:
                Rspec = self._compute_residuals(params_values, dlth, "TE")
                RlTE, WlTE = self._xspectra_to_xfreq(Rspec, self._dlweight["TE"], normed=False)
                Rl = Rl + RlTE
                Wl = Wl + WlTE
            if self._is_mode["ET"]:
                Rspec = self._compute_residuals(params_values, dlth, "ET")
                RlET, WlET = self._xspectra_to_xfreq(Rspec, self._dlweight["ET"], normed=False)
                Rl = Rl + RlET
                Wl = Wl + WlET
            # select multipole range
            Xl += self._select_spectra(Rl / Wl, 'TE')

        self.delta_cl = np.asarray(Xl).astype('float32')
#        chi2 = self.delta_cl @ self._invkll @ self.delta_cl
        chi2 = self._invkll.dot(self.delta_cl).dot(self.delta_cl)

        self.log.debug(f"chi2/ndof = {chi2}/{len(self.delta_cl)}")
        return chi2

    def get_requirements(self):
        return dict(Cl={mode: self.lmax for mode in ["tt", "ee", "te"]})

    def logp(self, **params_values):
        dl = self.theory.get_Cl(ell_factor=True)
        return self.loglike(dl, **params_values)

    def loglike(self, dl, **params_values):
        """
        Compute likelihood from model out of Boltzmann code
        Units: Dl in muK^2

        Parameters
        ----------
        pars: dict
              parameter values
        dl: dict
              CMB power spectrum (Dl in µK^2)

        Returns
        -------
        lnL: float
            Log likelihood for the given parameters -2ln(L)
        """
        # cl_boltz from Boltzmann (Cl in muK^2)
#        lth = np.arange(self.lmax + 1)
#        dlth = np.zeros((4, self.lmax + 1))
#        dlth["TT"] = dl["tt"][lth]
#        dlth["EE"] = dl["ee"][lth]
#        dlth["TE"] = dl["te"][lth]
        dlth = {k.upper():dl[k][:self.lmax+1] for k in dl.keys()}
        dlth['ET'] = dlth['TE']

        chi2 = self.compute_chi2(dlth, **params_values)

        return -0.5 * chi2

    @classmethod
    def get_path(cls, path):
        if path.rstrip(os.sep).endswith(data_path):
            return path
        return os.path.realpath(os.path.join(path, data_path))

    @classmethod
    def is_installed(cls, **kwargs):
        if kwargs.get("data", True):
            path = cls.get_path(kwargs["path"])
            if not (
                cls.get_install_options() and os.path.exists(path) and len(os.listdir(path)) > 0
            ):
                return False
            # Test if the covariance file is there
            test_path = os.path.join(path, f"**/*_{cls.__name__}.fits")
            return len(glob.glob(test_path, recursive=True)) > 0
        return True


# ------------------------------------------------------------------------------------------------


def _get_install_options(filename):
    return {"download_url": f"{data_url}/{filename}"}


class TTTEEE(_HillipopLikelihood):
    """High-L TT+TE+EE Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood
    with foreground models for cross-correlation spectra from Planck 100, 143 and 217 GHz
    split-frequency maps

    """

    install_options = _get_install_options("planck_2020_hillipop_TTTEEE_v4.2.tar.gz")


class TTTE(_HillipopLikelihood):
    """High-L TT+TE Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood
    with foreground models for cross-correlation spectra from Planck 100, 143 and 217 GHz
    split-frequency maps

    """

    install_options = _get_install_options("planck_2020_hillipop_TTTE_v4.2.tar.gz")


class TT(_HillipopLikelihood):
    """High-L TT Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217 GHz split-frequency
    maps

    """

    install_options = _get_install_options("planck_2020_hillipop_TT_v4.2.tar.gz")


class EE(_HillipopLikelihood):
    """High-L EE Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217 GHz split-frequency
    maps

    """

    install_options = _get_install_options("planck_2020_hillipop_EE_v4.2.tar.gz")


class TE(_HillipopLikelihood):
    """High-L TE Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217 GHz split-frequency
    maps

    """

    install_options = _get_install_options("planck_2020_hillipop_TE_v4.2.tar.gz")
