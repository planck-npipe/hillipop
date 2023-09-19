HiLLiPoP: High-L Likelihood Polarized for Planck
================================================
[![Unit test]( https://img.shields.io/github/actions/workflow/status/planck-npipe/hillipop/testing.yml?branch=master)](https://github.com/planck-npipe/hillipop/actions/workflows/testing.yml)
[![pypi](https://img.shields.io/pypi/v/planck-2020-hillipop)](https://pypi.python.org/pypi/planck-2020-hillipop)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


``Hillipop`` is a multifrequency CMB likelihood for Planck data. The likelihood is a spectrum-based
Gaussian approximation for cross-correlation spectra from Planck 100, 143 and 217GHz split-frequency
maps, with semi-analytic estimates of the Cl covariance matrix based on the data. The cross-spectra
are debiased from the effects of the mask and the beam leakage using
[``Xpol``](https://gitlab.in2p3.fr/tristram/Xpol) (a generalization to polarization of the algorithm
presented in [Tristram et al. 2005](https://arxiv.org/abs/astro-ph/0405575)) before being compared
to the model, which includes CMB and foreground residuals. They cover the multipoles from &ell;=30
to &ell;=2500.

The model consists of a linear combination of the CMB power spectrum and several foregrounds
residuals. These are:
- Galactic dust (estimated directly from the 353 GHz channel);
- the cosmic infrared background (as measured in [Planck Collaboration XXX
  2014](https://arxiv.org/abs/1309.0382));
- thermal Sunyaev-Zeldovich emission (based on the Planck measurement reported in [Planck
  Collaboration XXI 2014](https://arxiv.org/abs/1303.5081));
- kinetic Sunyaev-Zeldovich emission, including homogeneous and patchy reionization components from
  [Shaw et al. (2012)](https://arxiv.org/abs/1109.0553) and [Battaglia et
  al. (2013)](https://arxiv.org/abs/1211.2832);
- a tSZ-CIB correlation consistent with both models above; and
- unresolved point sources as a Poisson-like power spectrum.

HiLLiPoP has been used as an alternative to the public Planck likelihood in the 2013 and 2015 Planck
releases [[Planck Collaboration XV 2014](https://arxiv.org/abs/1303.5075); [Planck Collaboration XI
2016](https://arxiv.org/abs/1507.02704)], and is described in detail in [Couchot et
al. (2017)](https://arxiv.org/abs/1609.09730).

Likelihoods available are ``hillipop.TT``, ``hillipop.EE``, ``hillipop.TE``, ``hillipop.TTTE``, and
``hillipop.TTTEEE``.

It is interfaced with the ``cobaya`` MCMC sampler.

Likelihood versions
-------------------

<!-- * Planck 2018 (PR3) -->
* Planck 2020 (v4.1.0, PR4)

Install
-------
The easiest way to install the `Hillipop` likelihood is *via* `pip`

```shell
pip install planck-2020-hillipop [--user]
```

If you plan to dig into the code, it is better to clone this repository to some location

```shell
git clone https://github.com/planck-npipe/hillipop.git /where/to/clone
```

Then you can install the `Hillipop` likelihood and its dependencies *via*

```shell
pip install -e /where/to/clone
```

The ``-e`` option allow the developer to make changes within the `Hillipop` directory without having
to reinstall at every changes. If you plan to just use the likelihood and do not develop it, you can
remove the ``-e`` option.

Installing Hillipop likelihood data
-----------------------------------

The [`examples/hillipop_example.yaml`](examples/hillipop_example.yaml) file is a good starting point to
know the different nuisance parameters used by `hillipop` likelihoods.

You should use the `cobaya-install` binary to automatically download the data needed by the
`Hillipop` likelihood

```shell
cobaya-install /where/to/clone/examples/hillipop_example.yaml -p /where/to/put/packages
```

Data and code such as [CAMB](https://github.com/cmbant/CAMB) will be downloaded and installed within
the ``/where/to/put/packages`` directory. For more details, you can have a look to `cobaya`
[documentation](https://cobaya.readthedocs.io/en/latest/installation_cosmo.html).

Requirements
------------
* Python >= 3.5
* `numpy`
* `astropy`
