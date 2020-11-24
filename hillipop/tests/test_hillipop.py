import os
import tempfile
import unittest

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "Hillipop_packages"
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

calib_params = {
    "A_planck": 1.0,
    "cal100A": 0.0,
    "cal100B": 0.0,
    "cal143A": 0.0,
    "cal143B": 0.0,
    "cal217A": 0.0,
    "cal217B": 0.0,
}

nuisance_params = {
    "TT": {
        "Aradio": 1.0,
        "Adusty": 1.0,
        "AdustTT": 1.0,
        "Acib": 1.0,
        "Asz": 1.0,
        "Aksz": 1.0,
        "Aszxcib": 1.0,
    },
    "EE": {"AdustPP": 1.0},
    "TE": {"AdustTP": 1.0},
}
nuisance_params["TTTE"] = {
    **nuisance_params["TT"],
    **nuisance_params["TE"],
}
nuisance_params["TTTEEE"] = {
    **nuisance_params["TTTE"],
    **nuisance_params["EE"],
}

chi2s = {"TT": 12573.1, "EE": 7898.6, "TE": 8963.5}  # "TTTE": 22272.7, "TTTEEE": 31153.3}


class HillipopTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for mode in chi2s.keys():
            install(
                {"likelihood": {"hillipop.{}".format(mode): None}},
                path=packages_path,
                skip_global=True,
            )

    def test_hillipop(self):
        import camb
        import hillipop

        camb_cosmo = cosmo_params.copy()
        camb_cosmo.update({"lmax": 2500, "lens_potential_accuracy": 1})
        pars = camb.set_params(**camb_cosmo)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        cl_dict = {k: powers["total"][:, v] for k, v in {"tt": 0, "ee": 1, "te": 3}.items()}

        for mode, chi2 in chi2s.items():
            _hlp = getattr(hillipop, mode)
            my_lik = _hlp({"packages_path": packages_path})
            loglike = my_lik.loglike(cl_dict, **{**calib_params, **nuisance_params[mode]})
            self.assertAlmostEqual(-2 * loglike, chi2, 0)

    def test_cobaya(self):
        for mode, chi2 in chi2s.items():
            info = {
                "debug": True,
                "likelihood": {"hillipop.{}".format(mode): None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_params, **calib_params, **nuisance_params[mode]},
                "modules": packages_path,
            }
            from cobaya.model import get_model

            model = get_model(info)
            my_hillipop = model.likelihood["hillipop.{}".format(mode)]
            self.assertAlmostEqual(-2 * model.loglikes({})[0][0], chi2, 0)
