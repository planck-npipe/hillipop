import os
import tempfile
import unittest
import camb
from astropy.utils import minversion

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "Hillipop_packages"
)

cosmo_params = {
    "H0": 67.66,
    "As": 2.088434844099595e-09,
    "ombh2": 0.02226,
    "omch2": 0.1188,
    "ns": 0.9680,
    "tau": 0.0580,
}

calib_params = {
    "A_planck": 0.9996,
    "cal100A": 1.0087,
    "cal100B": 0.9993,
    "cal143A": 1.00,
    "cal143B": 1.0046,
    "cal217A": 0.9992,
    "cal217B": 1.0033,
    "pe100A": 1.0,
    "pe100B": 1.0,
    "pe143A": 1.0,
    "pe143B": 1.0,
    "pe217A": 0.975,
    "pe217B": 0.975
}

nuisance_params = {
    "TT": {
        "Aradio": 63.3,
        "Adusty": 6.11,
        "AdustT": 1.087,
        "beta_dustT": 1.513,
        "Acib": 0.99,
        "Atsz": 5.9,
        "Aksz": 1.0,
        "xi": 0.46,
        "beta_cib": 1.85,
        "beta_dusty": 1.85,
        "beta_radio": -0.8,
    },
    "EE": {
        "AdustP": 1.2,
        "beta_dustP": 1.595,
    },
    "TE": {
        "AdustT": 1.087,
        "beta_dustT": 1.513,
        "AdustP": 1.2,
        "beta_dustP": 1.595,
    },
}
nuisance_params["TTTE"] = {
    **nuisance_params["TT"],
    **nuisance_params["TE"],
}
nuisance_params["TTTEEE"] = {
    **nuisance_params["TTTE"],
    **nuisance_params["EE"],
}


if minversion(camb, "2.0.0"):
    chi2s = {"TT": 11066.50, "EE": 9326.64, "TE": 10077.94, "TT_bin": 1899.51, "TTTEEE_bin": 5501.55}
else:
    chi2s = {"TT": 11067.79, "EE": 9326.43, "TE": 10078.29, "TT_bin": 1900.86, "TTTEEE_bin": 5502.92}

class HillipopTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for mode in chi2s.keys():
            install(
                {"likelihood": {f"planck_2020_hillipop.{mode}": None}},
                path=packages_path,
                skip_global=True,
            )

    def test_hillipop(self):
        import camb
        import planck_2020_hillipop

        camb_cosmo = cosmo_params.copy()
        camb_cosmo.update({"lmax": 2500, "lens_potential_accuracy": 1})
        pars = camb.set_params(**camb_cosmo)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        cl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}

        for mode, chi2 in chi2s.items():
            _hlp = getattr(planck_2020_hillipop, mode)
            my_lik = _hlp({"packages_path": packages_path})
            loglike = my_lik.loglike(
                cl_dict,
                **{**calib_params, **nuisance_params[mode.replace("_bin", "")]},
            )
            self.assertLess(abs(-2 * loglike - chi2), 1)

    def test_cobaya(self):
        for mode, chi2 in chi2s.items():
            info = {
                "debug": True,
                "likelihood": {"planck_2020_hillipop.{}".format(mode): None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {
                    **cosmo_params,
                    **calib_params,
                    **nuisance_params[mode.replace("_bin", "")],
                },
                "packages_path": packages_path,
            }
            from cobaya.model import get_model

            model = get_model(info)
            self.assertLess(abs(-2 * model.loglikes({})[0][0] - chi2), 1)
