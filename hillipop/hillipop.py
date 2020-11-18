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
# TT
# ------------------------------------------------------------------------------------------------
class TT(_InstallableLikelihood):
    """High-L Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency
    maps

    """
    multipoles_range_file: Optional[str]
    frequencies: Optional[list]
    foregrounds: list

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
        
        #init likelihood
        self.hlp = tools.hlp_likelihood( self.multipoles_range_file,
                                         self.xspectra_basename,
                                         self.xspectra_errors_basename,
                                         self.covariance_matrix_basename,
                                         self.frequencies,
                                         self.foregrounds,
                                         self.log, data_folder=self.data_folder,
                                         TT=True, EE=False, TE=False, ET=False)

    def get_requirements(self):
        return dict(Cl={mode: self.lmax for mode in ["tt"]})

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
        lth = np.arange(self.hlp.lmax + 1)
        clth = np.zeros( (4,self.hlp.lmax+1) )
        clth[0] = cl["tt"][lth]

        chi2 = self.hlp.compute_chi2( clth, **params_values)

        return -0.5 * chi2






# ------------------------------------------------------------------------------------------------
# EE
# ------------------------------------------------------------------------------------------------
class EE(_InstallableLikelihood):
    """High-L Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency
    maps

    """
    multipoles_range_file: Optional[str]
    frequencies: Optional[list]
    foregrounds: list

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
        
        #init likelihood
        self.hlp = tools.hlp_likelihood( self.multipoles_range_file,
                                         self.xspectra_basename,
                                         self.xspectra_errors_basename,
                                         self.covariance_matrix_basename,
                                         self.frequencies,
                                         self.foregrounds,
                                         self.log, data_folder=self.data_folder,
                                         TT=False, EE=True, TE=False, ET=False)

    def get_requirements(self):
        return dict(Cl={mode: self.lmax for mode in ["ee"]})

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
        lth = np.arange(self.hlp.lmax + 1)
        clth = np.zeros( (4,self.hlp.lmax+1) )
        clth[1] = cl["ee"][lth]

        chi2 = self.hlp.compute_chi2( clth, **params_values)

        return -0.5 * chi2





# ------------------------------------------------------------------------------------------------
# TE
# ------------------------------------------------------------------------------------------------
class TE(_InstallableLikelihood):
    """High-L Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency
    maps

    """
    multipoles_range_file: Optional[str]
    frequencies: Optional[list]
    foregrounds: list

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
        
        #init likelihood
        self.hlp = tools.hlp_likelihood( self.multipoles_range_file,
                                         self.xspectra_basename,
                                         self.xspectra_errors_basename,
                                         self.covariance_matrix_basename,
                                         self.frequencies,
                                         self.foregrounds,
                                         self.log, data_folder=self.data_folder,
                                         TT=False, EE=False, TE=True, ET=True)

    def get_requirements(self):
        return dict(Cl={mode: self.lmax for mode in ["te"]})

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
        lth = np.arange(self.hlp.lmax + 1)
        clth = np.zeros( (4,self.hlp.lmax+1) )
        clth[3] = cl["te"][lth]

        chi2 = self.hlp.compute_chi2( clth, **params_values)

        return -0.5 * chi2




# ------------------------------------------------------------------------------------------------
# TT-EE-TE
# ------------------------------------------------------------------------------------------------
class TTTEEE(_InstallableLikelihood):
    """High-L Likelihood for Polarized Planck Spectra-based Gaussian-approximated likelihood with
    foreground models for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency
    maps

    """
    multipoles_range_file: Optional[str]
    frequencies: Optional[list]
    foregrounds: list

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
        
        #init likelihood
        self.hlp = tools.hlp_likelihood( self.multipoles_range_file,
                                         self.xspectra_basename,
                                         self.xspectra_errors_basename,
                                         self.covariance_matrix_basename,
                                         self.frequencies,
                                         self.foregrounds,
                                         self.log, data_folder=self.data_folder,
                                         TT=True, EE=True, TE=True, ET=True)

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
        lth = np.arange(self.hlp.lmax + 1)
        clth = np.zeros( (4,self.hlp.lmax+1) )
        clth[0] = cl["tt"][lth]
        clth[1] = cl["ee"][lth]
        clth[3] = cl["te"][lth]

        chi2 = self.hlp.compute_chi2( clth, **params_values)

        return -0.5 * chi2
