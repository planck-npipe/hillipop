import os
import tempfile
import unittest

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "LAT_packages"
)

cosmo_params = {
    "cosmomc_theta": 0.0104085,
    "As": 2.0989031673191437e-09,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "ns": 0.9649,
    "Alens": 1.0,
    "tau": 0.0544,
}

nuisance_params = {
    "Aplanck": 1.0,
    "Aradio": 1.61984,
    "Adusty": 0.781192,
    "AdustTT": 1.0,
    "AdustPP": 1.0,
    "AdustTP": 1.0,
    "Acib": 1.0,
    "Asz": 1.0,
    "Aksz": 0.6,
    "Aszxcib": 1.0,
    "c0": 0,
    "c1": 0,
    "c2": 0,
    "c3": 0,
    "c4": 0,
    "c5": 0,
}

chi2s = {"TT": 10347.50, "TE": 9332.91, "EE": 7327.07}


class HillipopTest(unittest.TestCase):
    # def setUp(self):
    #     from cobaya.install import install

    #     install({"likelihood": {"hillipop.Hillipop": None}}, path=packages_path)

    def test_hillipop(self):
        import camb

        camb_cosmo = cosmo_params.copy()
        camb_cosmo.update({"lmax": 2500, "lens_potential_accuracy": 1})
        pars = camb.set_params(**camb_cosmo)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        cl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}

        for mode, chi2 in chi2s.items():
            from hillipop import Hillipop

            my_lik = Hillipop(
                {
                    "packages_path": packages_path,
                    "TT": "TT" in mode,
                    "TE": "TE" in mode,
                    "ET": "TE" in mode,
                    "EE": "EE" in mode,
                }
            )

            loglike = my_lik.loglike(cl_dict, **nuisance_params)
            self.assertAlmostEqual(-2 * loglike, chi2, 1)

    def test_cobaya(self):
        info = {
            "debug": True,
            "likelihood": {
                "hillipop.Hillipop": {"TT": True, "EE": False, "TE": False, "ET": False}
            },
            "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
            "params": cosmo_params,
            "modules": packages_path,
        }
        from cobaya.model import get_model

        model = get_model(info)
        my_hillipop = model.likelihood["hillipop.Hillipop"]
        chi2 = -2 * model.loglikes(nuisance_params)[0]
        self.assertAlmostEqual(chi2[0], chi2s["TT"], 1)
