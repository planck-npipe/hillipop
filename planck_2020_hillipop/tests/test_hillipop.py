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
        "Aradio": 0.0,
        "Adusty": 0.0,
        "Ad100T": 0.016,
        "Ad143T": 0.035,
        "Ad217T": 0.130,
        "Acib": 1.0,
        "Asz": 1.0,
        "Aksz": 1.0,
        "Aszxcib": 1.0,
        "Aps_100x100": 283.,
        "Aps_100x143": 128.,
        "Aps_100x217": 90.,
        "Aps_143x143": 49.5,
        "Aps_143x217": 34.5,
        "Aps_217x217": 72.,
        },
    "EE": {
        "Ad100P": 0.016,
        "Ad143P": 0.035,
        "Ad217P": 0.130,
        },
    "TE": {
        "Ad100T": 0.02,
        "Ad143T": 0.04,
        "Ad217T": 0.13,
        "Ad100P": 0.016,
        "Ad143P": 0.035,
        "Ad217P": 0.130,
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
nuisance_equiv = {
    "Ad100": 0.016,
    "Ad143": 0.035,
    "Ad217": 0.130,
    }

chi2s = {"TT": 11415.58, "EE": 9244.86, "TE": 9916.65}


class HillipopTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        for mode in chi2s.keys():
            install(
                {"likelihood": {"planck_2020_hillipop.{}".format(mode): None}},
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
            loglike = my_lik.loglike(cl_dict, **{**calib_params, **nuisance_params[mode], **nuisance_equiv})
            self.assertLess( abs(-2 * loglike - chi2), 1)

    def test_cobaya(self):
        for mode, chi2 in chi2s.items():
            info = {
                "debug": True,
                "likelihood": {"planck_2020_hillipop.{}".format(mode): None},
                "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
                "params": {**cosmo_params, **calib_params, **nuisance_params[mode], **nuisance_equiv},
                "packages_path": packages_path,
            }
            from cobaya.model import get_model

            model = get_model(info)
            self.assertLess( abs(-2 * model.loglikes({})[0][0] - chi2), 1)
