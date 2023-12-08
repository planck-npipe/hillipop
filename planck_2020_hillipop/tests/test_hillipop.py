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
    "cal100A": 1.0,
    "cal100B": 1.0,
    "cal143A": 1.0,
    "cal143B": 1.0,
    "cal217A": 1.0,
    "cal217B": 1.0,
}

nuisance_params = {
    "TT": {
        "Aradio": 60.0,
        "Adusty": 6.0,
        "AdustT": 1.0,
        "beta_dustT": 1.51,
        "Acib": 1.0,
        "Atsz": 5.0,
        "Aksz": 0.1,
        "xi": 0.0,
        "beta_cib": 1.78,
        "beta_dusty": 1.78,
        "beta_radio": -0.8,
    },
    "EE": {
        "AdustP": 1.0,
        "beta_dustP": 1.59,
    },
    "TE": {
        "AdustT": 1.0,
        "beta_dustT": 1.51,
        "AdustP": 1.0,
        "beta_dustP": 1.59,
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
nuisance_equiv = {p: 1.0 for p in ["pe100A", "pe100B", "pe143A", "pe143B", "pe217A", "pe217B"]}

chi2s = {"TT": 11799.45, "EE": 9497.83, "TE": 10104.03, "TT_lite": 2636.29, "TTTEEE_lite": 6421.71}


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
                **{**calib_params, **nuisance_params[mode.replace("_lite", "")], **nuisance_equiv},
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
                    **nuisance_params[mode.replace("_lite", "")],
                    **nuisance_equiv,
                },
                "packages_path": packages_path,
            }
            from cobaya.model import get_model

            model = get_model(info)
            self.assertLess(abs(-2 * model.loglikes({})[0][0] - chi2), 1)
