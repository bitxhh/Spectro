"""Тесты модуля spectrolib.physics."""

import numpy as np
import pytest

from spectrolib.physics import (
    nm_to_wavenumber, wavenumber_to_nm,
    ppm_to_fraction, fraction_to_ppm,
    number_density, beer_lambert,
)


class TestUnits:

    def test_nm_wavenumber_roundtrip(self):
        wl = np.array([200.0, 500.0, 1000.0, 5000.0])
        assert np.allclose(wavenumber_to_nm(nm_to_wavenumber(wl)), wl)

    def test_nm_to_wavenumber_known(self):
        # 1000 нм → 10000 см⁻¹
        assert nm_to_wavenumber(1000.0) == pytest.approx(10000.0, rel=1e-10)
        # 500 нм → 20000 см⁻¹
        assert nm_to_wavenumber(500.0) == pytest.approx(20000.0, rel=1e-10)

    def test_ppm_fraction_roundtrip(self):
        c = np.array([1.0, 100.0, 1e6])
        assert np.allclose(fraction_to_ppm(ppm_to_fraction(c)), c)


class TestNumberDensity:

    def test_loschmidt(self):
        """При 273.15 K и 1 атм должно быть число Лошмидта ≈ 2.687e19 см⁻³."""
        n = number_density(273.15, 1.0)
        assert n == pytest.approx(2.687e19, rel=1e-3)

    def test_proportionality_in_pressure(self):
        n1 = number_density(310.0, 1.0)
        n2 = number_density(310.0, 2.0)
        assert n2 == pytest.approx(2 * n1, rel=1e-10)

    def test_inverse_in_temperature(self):
        n1 = number_density(300.0, 1.0)
        n2 = number_density(600.0, 1.0)
        assert n2 == pytest.approx(0.5 * n1, rel=1e-10)


class TestBeerLambert:

    def test_linearity_in_concentration(self):
        """OD должна быть линейна по концентрации (БЛБ — линейный закон)."""
        sigma = np.array([1e-20, 2e-20, 3e-20])  # см²/молекула
        od1 = beer_lambert(sigma, c_ppm=1.0, L_cm=10.0)
        od2 = beer_lambert(sigma, c_ppm=2.0, L_cm=10.0)
        assert np.allclose(od2, 2 * od1)

    def test_linearity_in_path(self):
        sigma = np.array([1e-20])
        od1 = beer_lambert(sigma, c_ppm=100.0, L_cm=1.0)
        od2 = beer_lambert(sigma, c_ppm=100.0, L_cm=10.0)
        assert np.allclose(od2, 10 * od1)

    def test_dimensional_sanity(self):
        """OD должна быть безразмерной и порядка единицы для типичных
        биомаркеров: σ ~ 1e-19 см², L = 10 см, c = 1 ppm, p=1 атм."""
        sigma = 1e-19
        od = beer_lambert(np.array([sigma]), c_ppm=1.0, L_cm=10.0,
                          T_K=296, p_atm=1.0)
        # N_total(296, 1) ≈ 2.48e19, N_target = 1e-6 * 2.48e19 = 2.48e13
        # OD = 1e-19 * 2.48e13 * 10 = 2.48e-5
        assert od[0] == pytest.approx(2.48e-5, rel=1e-2)
