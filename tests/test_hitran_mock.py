"""
Тесты интеграции с HITRAN — через мок hapi (без сетевых вызовов).

Цель: проверить, что add_molecule корректно:
1. Передаёт правильные параметры в hapi (диапазон ν, T, p, диluent).
2. Применяет БЛБ к возвращённому сечению σ.
3. Интерполирует с тонкой ν-сетки hapi на нашу λ-сетку без потерь и
   с правильным переворотом порядка (нм возрастает ⇄ см⁻¹ убывает).

Реальные тесты против HITRAN (с сетевым вызовом) — отдельным маркером
и в этом файле их нет.
"""

import numpy as np
import pytest
from unittest.mock import patch

from spectrolib import Spectrum
from spectrolib.physics import number_density


def _fake_voigt(SourceTables, Environment, WavenumberRange, WavenumberStep,
                WavenumberWing, Diluent, HITRAN_units):
    """
    Мок hapi.absorptionCoefficient_Voigt.

    Возвращает константное сечение σ₀ = 1e-20 см²/молекула на сетке.
    Это позволяет аналитически проверить OD-результат.
    """
    nu_min, nu_max = WavenumberRange
    nu_grid = np.arange(nu_min, nu_max + WavenumberStep, WavenumberStep)
    sigma = np.full_like(nu_grid, 1e-20)
    return nu_grid, sigma


def _fake_voigt_with_peak(SourceTables, Environment, WavenumberRange,
                          WavenumberStep, WavenumberWing, Diluent,
                          HITRAN_units):
    """
    Мок с одной гауссовой 'линией' посредине — для проверки интерполяции.
    """
    nu_min, nu_max = WavenumberRange
    nu_grid = np.arange(nu_min, nu_max + WavenumberStep, WavenumberStep)
    nu_center = 0.5 * (nu_min + nu_max)
    sigma = 1e-19 * np.exp(-((nu_grid - nu_center) / 0.1) ** 2)
    return nu_grid, sigma


@pytest.fixture(autouse=True)
def patch_fetch():
    """Подменяем сетевой вызов fetch_molecule на тривиальный no-op."""
    with patch('spectrolib.spectrum.fetch_molecule',
                return_value='FAKE_TABLE'):
        with patch('spectrolib.spectrum.init_db'):
            yield


class TestBeerLambertIntegration:

    def test_constant_sigma_gives_expected_OD(self):
        """
        При постоянном σ = 1e-20 см²/молекула:
            OD = σ · N_target · L
            N_target = ppm·1e-6 · N_total(T, p)
        Проверяем точное численное совпадение.
        """
        with patch('hapi.absorptionCoefficient_Voigt', side_effect=_fake_voigt):
            spec = Spectrum.from_range(750, 770, step_nm=0.01)
            spec.add_molecule('O2', c_ppm=1000, L_cm=10,
                              T_K=296, p_atm=1.0)

        # Ожидаемое OD
        sigma = 1e-20
        N_total = number_density(296, 1.0)
        N_target = 1000 * 1e-6 * N_total
        expected_OD = sigma * N_target * 10

        # Должно быть константой по всей сетке (с учётом интерполяции)
        assert np.allclose(spec.true_optical_depth, expected_OD, rtol=1e-6)

    def test_linearity_in_concentration(self):
        with patch('hapi.absorptionCoefficient_Voigt', side_effect=_fake_voigt):
            s1 = Spectrum.from_range(750, 770, step_nm=0.01)
            s1.add_molecule('O2', c_ppm=100, L_cm=10)
            s2 = Spectrum.from_range(750, 770, step_nm=0.01)
            s2.add_molecule('O2', c_ppm=200, L_cm=10)
        assert np.allclose(s2.true_optical_depth,
                           2 * s1.true_optical_depth, rtol=1e-6)

    def test_additivity_of_molecules(self):
        """OD двух молекул = OD₁ + OD₂."""
        with patch('hapi.absorptionCoefficient_Voigt', side_effect=_fake_voigt):
            s_both = Spectrum.from_range(750, 770, step_nm=0.01)
            s_both.add_molecule('O2', c_ppm=100, L_cm=10)
            s_both.add_molecule('H2O', c_ppm=200, L_cm=10)

            s_o2 = Spectrum.from_range(750, 770, step_nm=0.01)
            s_o2.add_molecule('O2', c_ppm=100, L_cm=10)

            s_h2o = Spectrum.from_range(750, 770, step_nm=0.01)
            s_h2o.add_molecule('H2O', c_ppm=200, L_cm=10)

        assert np.allclose(s_both.true_optical_depth,
                           s_o2.true_optical_depth + s_h2o.true_optical_depth,
                           rtol=1e-6)


class TestInterpolation:

    def test_peak_position_preserved(self):
        """После интерполяции с ν-сетки на λ-сетку положение пика не съезжает."""
        # Центр диапазона по ν соответствует центру по λ только примерно
        # (нелинейная связь). Берём узкий диапазон, чтобы нелинейность
        # была пренебрежима.
        with patch('hapi.absorptionCoefficient_Voigt',
                   side_effect=_fake_voigt_with_peak):
            spec = Spectrum.from_range(759.9, 760.1, step_nm=0.0001)
            spec.add_molecule('O2', c_ppm=1000, L_cm=10)

        idx_max = int(np.argmax(spec.true_optical_depth))
        peak_lambda = spec.wavelength_nm[idx_max]
        # Центр диапазона ν соответствует центру λ ≈ 760 нм
        assert peak_lambda == pytest.approx(760.0, abs=0.005)

    def test_grid_orientation_correct(self):
        """
        Внутренний переворот: hapi даёт ν возрастающую, наша сетка по λ
        возрастает, что = ν убывает. Проверяем, что интерполяция не
        перепутала направление.

        Используем «асимметричный» мок: σ растёт с ν.
        """
        def asymmetric(SourceTables, Environment, WavenumberRange,
                       WavenumberStep, WavenumberWing, Diluent,
                       HITRAN_units):
            nu_min, nu_max = WavenumberRange
            nu_grid = np.arange(nu_min, nu_max + WavenumberStep,
                                 WavenumberStep)
            # σ линейно растёт с ν → линейно растёт с уменьшением λ
            sigma = 1e-20 * (nu_grid - nu_min) / (nu_max - nu_min)
            return nu_grid, sigma

        with patch('hapi.absorptionCoefficient_Voigt', side_effect=asymmetric):
            spec = Spectrum.from_range(750, 770, step_nm=0.01)
            spec.add_molecule('O2', c_ppm=10000, L_cm=10)

        # OD должна убывать с ростом λ (потому что σ растёт с ν, а ν↔1/λ)
        od = spec.true_optical_depth
        # Тренд (грубо): первая четверть должна быть выше последней
        assert od[: len(od) // 4].mean() > od[-len(od) // 4 :].mean()


class TestTemperatureForwardedToHapi:
    """
    Проверка: T_K из GasMixture/add_molecule долетает до hapi через
    Environment={'T': T_K, 'p': p_atm}. Это и есть «температурная
    коррекция сечений HITRAN»: hapi внутри пересчитывает S(T) от
    опорной T_REF_HITRAN_K=296 K через Q(T)/Q(T_ref).
    """

    def test_T_K_forwarded_to_hapi_environment(self):
        captured = {}

        def capture(SourceTables, Environment, WavenumberRange,
                    WavenumberStep, WavenumberWing, Diluent,
                    HITRAN_units):
            captured['Environment'] = dict(Environment)
            return _fake_voigt(SourceTables, Environment, WavenumberRange,
                                WavenumberStep, WavenumberWing, Diluent,
                                HITRAN_units)

        with patch('hapi.absorptionCoefficient_Voigt', side_effect=capture):
            spec = Spectrum.from_range(750, 770, step_nm=0.01)
            spec.add_molecule('O2', c_ppm=1000, L_cm=10,
                              T_K=310.0, p_atm=1.0)

        assert captured['Environment']['T'] == 310.0
        assert captured['Environment']['p'] == 1.0

    def test_T_REF_constant_exported(self):
        from spectrolib import T_REF_HITRAN_K
        assert T_REF_HITRAN_K == 296.0

    def test_different_T_K_passed_through(self):
        """При разных T_K hapi должен получить именно эту T (296 vs 310)."""
        calls = []

        def capture(SourceTables, Environment, WavenumberRange,
                    WavenumberStep, WavenumberWing, Diluent,
                    HITRAN_units):
            calls.append(Environment['T'])
            return _fake_voigt(SourceTables, Environment, WavenumberRange,
                                WavenumberStep, WavenumberWing, Diluent,
                                HITRAN_units)

        with patch('hapi.absorptionCoefficient_Voigt', side_effect=capture):
            s1 = Spectrum.from_range(750, 770, step_nm=0.01)
            s1.add_molecule('O2', c_ppm=1000, L_cm=10, T_K=296.0)
            s2 = Spectrum.from_range(750, 770, step_nm=0.01)
            s2.add_molecule('O2', c_ppm=1000, L_cm=10, T_K=310.0)

        assert calls == [296.0, 310.0]


class TestMetadataAfterHITRAN:

    def test_molecule_record_complete(self):
        with patch('hapi.absorptionCoefficient_Voigt', side_effect=_fake_voigt):
            spec = Spectrum.from_range(750, 770, step_nm=0.01)
            spec.add_molecule('CO2', c_ppm=4e5, L_cm=10,
                              T_K=310, p_atm=1.0,
                              diluent={'air': 0.96, 'self': 0.04})
        assert len(spec.molecules) == 1
        m = spec.molecules[0]
        assert m['name'] == 'CO2'
        assert m['c_ppm'] == 4e5
        assert m['L_cm'] == 10
        assert m['T_K'] == 310
        assert m['p_atm'] == 1.0
        assert m['diluent'] == {'air': 0.96, 'self': 0.04}
        assert m['profile'] == 'voigt'
