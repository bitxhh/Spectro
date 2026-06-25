"""
Тест согласованности баз данных сечений на перекрытии диапазонов.

Требование диплома: три источника σ — HITRAN (line-by-line), PNNL (ИК VOC) и
MPI-Mainz (УФ/ВИД) — должны выдавать СОГЛАСОВАННЫЕ сечения. Согласованность
обеспечивается тем, что независимо от источника:

  * σ приводится к ОДНОЙ единице — см²/молекула;
  * дальше всё идёт по ОДНОМУ хвосту Spectrum._accumulate_sigma:
        OD = σ · N(T,p) · L,  затем интерполяция на сетку спектра.

Поэтому если подать в HITRAN-путь и в PNNL/MPI-путь ОДНО И ТО ЖЕ эталонное
σ(ν), итоговая оптическая плотность на перекрытии диапазонов обязана совпасть.
Именно это здесь и проверяется (PNNL↔HITRAN в ИК, MPI↔HITRAN в УФ), плюс
отдельно — линейная интерполяция PNNL по T строго до 310 K.

Реальные файлы PNNL/MPI у пользователя пока отсутствуют (Zenodo/Globus), поэтому
данные синтезируются из аналитического эталонного σ(ν) и пишутся во временные
файлы в формате соответствующей базы. Это тестирует именно ПЛУМБИНГ
согласованности (общая единица + общая формула OD), а не числа конкретных
молекул.
"""

import numpy as np
import pytest
from unittest.mock import patch

from spectrolib import Spectrum
from spectrolib import pnnl, mpi
from spectrolib.physics import number_density, nm_to_wavenumber
from spectrolib.pnnl import (
    pnnl_absorbance_to_sigma,
    PNNL_PPM, PNNL_PATH_CM, T_REF_PNNL_K, P_REF_PNNL_ATM, T_EXHALE_K,
)

_LN10 = np.log(10.0)


# ---------------------------------------------------------------------------
# Эталонное σ(ν) и запись синтетических файлов баз
# ---------------------------------------------------------------------------

def _reference_sigma(nu_cm, nu0, width, peak=1e-19, base=1e-21):
    """Гладкое эталонное сечение [см²/молекула] как функция ν [см⁻¹].

    Широкий гауссов пик + база. Гладкость нужна, чтобы интерполяция с разных
    исходных сеток (PNNL-файл vs мок-HITRAN) на сетку спектра давала почти
    одинаковый результат.
    """
    nu_cm = np.asarray(nu_cm, dtype=float)
    return base + peak * np.exp(-((nu_cm - nu0) / width) ** 2)


def _pnnl_column_density():
    """Колоночная плотность нормировки PNNL: 1 ppm · 1 м при 296 K, 1 атм."""
    return PNNL_PPM * number_density(T_REF_PNNL_K, P_REF_PNNL_ATM) * PNNL_PATH_CM


def _write_pnnl_file(path, nu_cm, sigma):
    """Записать PNNL-файл (ν[см⁻¹], absorbance) из заданного σ.

    absorbance = σ · column_density / ln(10)  — точная обратная к
    pnnl_absorbance_to_sigma, так что загрузчик вернёт ровно это σ.
    """
    absorbance = np.asarray(sigma, dtype=float) * _pnnl_column_density() / _LN10
    lines = ["# synthetic PNNL test file  nu[cm-1]  absorbance(ppm-m, base10)"]
    for x, a in zip(nu_cm, absorbance):
        lines.append(f"{x:.6f}\t{a:.8e}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _write_mpi_file(path, lambda_nm, sigma):
    """Записать MPI-файл (λ[нм], σ[см²])."""
    lines = ["# synthetic MPI test file  lambda[nm,air]  sigma[cm2]"]
    for lam, s in zip(lambda_nm, sigma):
        lines.append(f"{lam:.6f}\t{s:.8e}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_registries():
    """Чистим in-memory индексы баз до и после каждого теста."""
    pnnl.clear_cache()
    mpi.clear_cache()
    yield
    pnnl.clear_cache()
    mpi.clear_cache()


@pytest.fixture
def patch_hitran():
    """Заглушки сетевых вызовов HITRAN (как в test_hitran_mock)."""
    with patch('spectrolib.spectrum.fetch_molecule', return_value='FAKE_TABLE'):
        with patch('spectrolib.spectrum.init_db'):
            yield


# ---------------------------------------------------------------------------
# 1) Перевод PNNL absorbance → σ
# ---------------------------------------------------------------------------

class TestPnnlAbsorbanceToSigma:

    def test_roundtrip_sigma(self):
        """σ → absorbance → σ восстанавливается точно."""
        sigma = np.array([1e-21, 5e-20, 1e-19, 3e-19])
        absorbance = sigma * _pnnl_column_density() / _LN10
        back = pnnl_absorbance_to_sigma(absorbance)
        assert np.allclose(back, sigma, rtol=1e-12)

    def test_numeric_scale(self):
        """σ ≈ A · 9.3e-16 (порядок из документации PNNL)."""
        sigma_per_unit_A = pnnl_absorbance_to_sigma(1.0)
        assert sigma_per_unit_A == pytest.approx(9.3e-16, rel=0.1)


# ---------------------------------------------------------------------------
# 2) Линейная интерполяция PNNL по T строго до 310 K
# ---------------------------------------------------------------------------

class TestPnnlTemperatureInterpolation:

    def _register_two_temps(self, tmp_path, peak_lo, peak_hi):
        """Зарегистрировать два PNNL-файла при 298.15 и 323.15 K.

        Сечение при низкой/высокой T различается по амплитуде пика — так
        интерполяция по каждой точке даёт нетривиальный, проверяемый результат.
        """
        nu = np.linspace(2900.0, 3100.0, 2001)   # общая плотная сетка
        sig_lo = _reference_sigma(nu, nu0=3000.0, width=20.0, peak=peak_lo)
        sig_hi = _reference_sigma(nu, nu0=3000.0, width=20.0, peak=peak_hi)
        p_lo = _write_pnnl_file(tmp_path / "TESTIR_298K.txt", nu, sig_lo)
        p_hi = _write_pnnl_file(tmp_path / "TESTIR_323K.txt", nu, sig_hi)
        pnnl.register_pnnl_files('TESTIR', {298.15: p_lo, 323.15: p_hi})
        return nu, sig_lo, sig_hi

    def test_default_target_is_310(self, tmp_path):
        """По умолчанию интерполяция идёт строго на 310 K (температура выдоха)."""
        assert T_EXHALE_K == 310.0
        self._register_two_temps(tmp_path, peak_lo=1e-19, peak_hi=2e-19)
        _, _, meta = pnnl.load_pnnl_sigma('TESTIR')
        assert meta['T_target_K'] == 310.0
        assert meta['mode'] == 'interp'
        assert meta['T_used'] == (298.15, 323.15)

    def test_per_point_linear_interpolation(self, tmp_path):
        """В каждой точке сетки: σ(310) = σ_lo + (σ_hi−σ_lo)·w."""
        nu, sig_lo, sig_hi = self._register_two_temps(
            tmp_path, peak_lo=1e-19, peak_hi=3e-19)
        nu_out, sig_out, _ = pnnl.load_pnnl_sigma('TESTIR', T_target=310.0)

        w = (310.0 - 298.15) / (323.15 - 298.15)
        expected = np.interp(nu_out, nu, sig_lo) + \
            (np.interp(nu_out, nu, sig_hi) - np.interp(nu_out, nu, sig_lo)) * w
        assert np.allclose(sig_out, expected, rtol=1e-10)

    def test_exact_temperature_no_interp(self, tmp_path):
        """Если есть файл ровно при T_target — берётся он без интерполяции."""
        nu = np.linspace(2900.0, 3100.0, 501)
        sig = _reference_sigma(nu, nu0=3000.0, width=20.0)
        p = _write_pnnl_file(tmp_path / "TESTIR_310K.txt", nu, sig)
        pnnl.register_pnnl_files('TESTIR', {310.0: p})
        _, sig_out, meta = pnnl.load_pnnl_sigma('TESTIR', T_target=310.0)
        assert meta['mode'] == 'exact'
        assert np.allclose(sig_out, sig, rtol=1e-10)

    def test_extrapolation_guarded(self, tmp_path):
        """Запрос вне [298.15, 323.15] K по умолчанию — ошибка (не экстраполяция)."""
        self._register_two_temps(tmp_path, peak_lo=1e-19, peak_hi=2e-19)
        with pytest.raises(ValueError):
            pnnl.load_pnnl_sigma('TESTIR', T_target=350.0)

    def test_extrapolation_clamped_when_allowed(self, tmp_path):
        """allow_extrapolation=True зажимает к крайней доступной T."""
        self._register_two_temps(tmp_path, peak_lo=1e-19, peak_hi=2e-19)
        _, _, meta = pnnl.load_pnnl_sigma(
            'TESTIR', T_target=350.0, allow_extrapolation=True)
        assert meta['mode'] == 'clamped'
        assert meta['T_used'] == (323.15,)


# ---------------------------------------------------------------------------
# 3) PNNL ↔ HITRAN: согласованность на перекрытии (ИК)
# ---------------------------------------------------------------------------

class TestPnnlHitranOverlap:

    NU0 = 3000.0      # центр пика, см⁻¹  (≈ 3333 нм)
    WIDTH = 25.0
    WL_MIN, WL_MAX = 3300.0, 3367.0   # нм, узкое окно вокруг пика

    def _mock_voigt(self):
        nu0, width = self.NU0, self.WIDTH

        def fake(SourceTables, Environment, WavenumberRange, WavenumberStep,
                 WavenumberWing, Diluent, HITRAN_units):
            nu_min, nu_max = WavenumberRange
            nu_grid = np.arange(nu_min, nu_max + WavenumberStep, WavenumberStep)
            return nu_grid, _reference_sigma(nu_grid, nu0, width)
        return fake

    def test_same_sigma_gives_same_OD(self, tmp_path, patch_hitran):
        """Одно и то же эталонное σ через PNNL-путь и HITRAN-путь → одинаковая OD."""
        # PNNL-файл: плотная сетка, шире окна спектра (чтобы края не занулялись).
        nu_file = np.linspace(2950.0, 3050.0, 4001)
        sig_file = _reference_sigma(nu_file, self.NU0, self.WIDTH)
        p = _write_pnnl_file(tmp_path / "TESTIR_310K.txt", nu_file, sig_file)
        pnnl.register_pnnl_files('TESTIR', {310.0: p})

        common = dict(c_ppm=5000.0, L_cm=10.0, T_K=310.0, p_atm=1.0)

        s_pnnl = Spectrum.from_range(self.WL_MIN, self.WL_MAX, step_nm=0.02)
        s_pnnl.add_molecule('TESTIR', source='pnnl', **common)

        with patch('hapi.absorptionCoefficient_Voigt',
                   side_effect=self._mock_voigt()):
            s_hit = Spectrum.from_range(self.WL_MIN, self.WL_MAX, step_nm=0.02)
            s_hit.add_molecule('TESTIR', source='hitran', **common)

        od_p = s_pnnl.true_optical_depth
        od_h = s_hit.true_optical_depth

        # Сравниваем там, где сигнал заметен (вне крыльев).
        mask = od_h > 0.05 * od_h.max()
        assert mask.sum() > 10
        assert np.allclose(od_p[mask], od_h[mask], rtol=2e-2)

    def test_loaded_pnnl_sigma_matches_reference(self, tmp_path):
        """Загруженное PNNL σ численно совпадает с эталоном (общая единица см²)."""
        nu_file = np.linspace(2950.0, 3050.0, 2001)
        sig_file = _reference_sigma(nu_file, self.NU0, self.WIDTH)
        p = _write_pnnl_file(tmp_path / "TESTIR_310K.txt", nu_file, sig_file)
        pnnl.register_pnnl_files('TESTIR', {310.0: p})
        nu_out, sig_out, _ = pnnl.load_pnnl_sigma('TESTIR', T_target=310.0)
        assert np.allclose(sig_out, _reference_sigma(nu_out, self.NU0, self.WIDTH),
                           rtol=1e-9)


# ---------------------------------------------------------------------------
# 4) MPI ↔ HITRAN: согласованность на перекрытии (УФ)
# ---------------------------------------------------------------------------

class TestMpiHitranOverlap:

    WL_MIN, WL_MAX = 250.0, 260.0     # нм (УФ)
    # центр пика в ν, соответствующий ~255 нм
    NU0 = nm_to_wavenumber(255.0)
    WIDTH = 400.0                      # см⁻¹

    def _mock_voigt(self):
        nu0, width = self.NU0, self.WIDTH

        def fake(SourceTables, Environment, WavenumberRange, WavenumberStep,
                 WavenumberWing, Diluent, HITRAN_units):
            nu_min, nu_max = WavenumberRange
            nu_grid = np.arange(nu_min, nu_max + WavenumberStep, WavenumberStep)
            return nu_grid, _reference_sigma(nu_grid, nu0, width)
        return fake

    def test_same_sigma_gives_same_OD(self, tmp_path, patch_hitran):
        """Эталонное σ через MPI-путь и HITRAN-путь → одинаковая OD (УФ)."""
        # MPI-файл: λ[нм] шире окна спектра, σ = эталон(ν).
        lam_file = np.linspace(248.0, 262.0, 4001)
        nu_file = nm_to_wavenumber(lam_file)
        sig_file = _reference_sigma(nu_file, self.NU0, self.WIDTH)
        p = _write_mpi_file(tmp_path / "TESTUV_295K_248-262nm.txt",
                            lam_file, sig_file)
        mpi.register_mpi_file('TESTUV', p, T_K=295.0)

        common = dict(c_ppm=5000.0, L_cm=10.0, T_K=296.0, p_atm=1.0)

        s_mpi = Spectrum.from_range(self.WL_MIN, self.WL_MAX, step_nm=0.005)
        s_mpi.add_molecule('TESTUV', source='mpi', **common)

        with patch('hapi.absorptionCoefficient_Voigt',
                   side_effect=self._mock_voigt()):
            s_hit = Spectrum.from_range(self.WL_MIN, self.WL_MAX, step_nm=0.005)
            s_hit.add_molecule('TESTUV', source='hitran', **common)

        od_m = s_mpi.true_optical_depth
        od_h = s_hit.true_optical_depth

        mask = od_h > 0.05 * od_h.max()
        assert mask.sum() > 10
        assert np.allclose(od_m[mask], od_h[mask], rtol=2e-2)


# ---------------------------------------------------------------------------
# 5) Маршрутизация источника (явный source= перекрывает реестр)
# ---------------------------------------------------------------------------

class TestSourceRouting:

    def test_explicit_source_overrides_registry(self, tmp_path, patch_hitran):
        """Явный source= для молекулы из реестра уводит в HITRAN-путь."""
        from spectrolib import resolve_source, MOLECULE_SOURCE
        # acetone в реестре → hitran_xsc
        assert MOLECULE_SOURCE.get('acetone') == 'hitran_xsc'
        assert resolve_source('acetone') == 'hitran_xsc'
        # явный source перекрывает
        assert resolve_source('acetone', source='hitran') == 'hitran'
