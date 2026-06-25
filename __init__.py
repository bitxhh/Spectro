"""
spectrolib — библиотека для синтеза и обработки спектров.

Два стиля API, можно смешивать:

1) Fluent (для интерактивной отладки в ноутбуке):

    from spectrolib import Spectrum, GaussILS, NoiseModel

    spec = (Spectrum.from_range(750, 770, step_nm=0.001)
            .add_molecule('O2', c_ppm=210000, L_cm=10)
            .add_molecule('H2O', c_ppm=10000, L_cm=10)
            .convolve_ils(GaussILS(fwhm=1.0))
            .add_noise_model(NoiseModel(thermal_sigma=0.005), seed=42))

    plt.plot(spec.wavelength_nm, spec.transmittance)
    plt.plot(spec.wavelength_nm, spec.true_transmittance, '--')

2) Объектный (для пайплайнов и сценариев заболеваний):

    from spectrolib import Instrument, GasMixture, NoiseModel, SpectrumGenerator
    from spectrolib import GaussILS

    inst = Instrument(wavelength_range=(750, 770), sampling_step=0.001,
                      ils=GaussILS(fwhm=1.0))
    mix = GasMixture(composition={'O2': 210000, 'H2O': 10000})
    noise = NoiseModel(thermal_sigma=0.005, shot_n_photons_max=1e4)

    gen = SpectrumGenerator(inst, noise_model=noise, seed=42)
    spec = gen.generate(mix)
"""

# Основной класс
from .spectrum import Spectrum

# ILS
from .ils import (
    ILS,
    GaussILS,
    LorentzILS,
    VoigtILS,
    FromFileILS,
    fwhm_to_sigma,
    sigma_to_fwhm,
    gauss_convolve,   # обратная совместимость
)

# Шумы
from .noise import NoiseModel, load_noise_model

# Объектный фасад
from .api import Instrument, GasMixture, SpectrumGenerator, load_instrument

# Физика
from .physics import (
    nm_to_wavenumber,
    wavenumber_to_nm,
    number_density,
    ppm_to_fraction,
    fraction_to_ppm,
    beer_lambert,
)

# Работа с HITRAN и I/O
from .hitran import fetch_molecule, list_local_tables, MOLECULE_IDS, T_REF_HITRAN_K
from .io import load_spectrum, save_spectrum

# Реестр источников сечений (HITRAN / PNNL / MPI / HITRAN xsc)
from .databases import (
    DB_HITRAN, DB_PNNL, DB_MPI, DB_HITRAN_XSC,
    MOLECULE_SOURCE, register_molecule, resolve_source,
)
# Доп. базы данных сечений
from . import pnnl, mpi, hitran_xsc

# Этапы пробоподготовки / протокол измерения
from .protocol import preconcentrate

# Визуализация
from .plotting import plot_spectrum, plot_clean_vs_noisy, plot_overlay, plot_snr_vs_n
from . import plotstyle  # отдельный модуль с едиными дефолтами/палитрой

# Биомаркерные панели
from .panels import MixturePanel, Biomarker, load_mixture_panel

# Многоканальная регистрация
from .channels import (
    Channel, ChannelSet, ChannelizedSpectrum,
    channelize, load_channel_set,
)

# Покомпонентный информационный аудит (глава 4 диплома)
from .audit import (
    LocalBlock, AuditResult,
    audit_block, design_concentrations, lognormal_percentile,
    compute_J_star_min, compute_kappa_max, round_kappa_floor,
    build_response_matrix, build_noise_covariance_OD,
    build_drift_basis, oblique_projector_perp, marginal_fisher,
    compute_J_star_m, population_params,
    plot_audit_trajectory, plot_audit_configuration,
)


__all__ = [
    # Основной класс
    'Spectrum',
    # ILS
    'ILS', 'GaussILS', 'LorentzILS', 'VoigtILS', 'FromFileILS',
    'fwhm_to_sigma', 'sigma_to_fwhm', 'gauss_convolve',
    # Шумы
    'NoiseModel', 'load_noise_model',
    # Объектный фасад
    'Instrument', 'GasMixture', 'SpectrumGenerator', 'load_instrument',
    # Физика
    'nm_to_wavenumber', 'wavenumber_to_nm', 'number_density',
    'ppm_to_fraction', 'fraction_to_ppm', 'beer_lambert',
    # HITRAN и I/O
    'fetch_molecule', 'list_local_tables', 'MOLECULE_IDS', 'T_REF_HITRAN_K',
    'load_spectrum', 'save_spectrum',
    # Реестр источников сечений и доп. базы данных
    'DB_HITRAN', 'DB_PNNL', 'DB_MPI', 'DB_HITRAN_XSC',
    'MOLECULE_SOURCE', 'register_molecule', 'resolve_source',
    'pnnl', 'mpi', 'hitran_xsc',
    # Протокол измерения
    'preconcentrate',
    # Визуализация
    'plot_spectrum', 'plot_clean_vs_noisy', 'plot_overlay', 'plot_snr_vs_n',
    'plotstyle',
    # Биомаркерные панели
    'MixturePanel', 'Biomarker', 'load_mixture_panel',
    # Многоканальная регистрация
    'Channel', 'ChannelSet', 'ChannelizedSpectrum',
    'channelize', 'load_channel_set',
    # Покомпонентный аудит
    'LocalBlock', 'AuditResult',
    'audit_block', 'design_concentrations', 'lognormal_percentile',
    'compute_J_star_min', 'compute_kappa_max', 'round_kappa_floor',
    'build_response_matrix', 'build_noise_covariance_OD',
    'build_drift_basis', 'oblique_projector_perp', 'marginal_fisher',
    'compute_J_star_m', 'population_params',
    'plot_audit_trajectory', 'plot_audit_configuration',
]

__version__ = '0.8.0'
