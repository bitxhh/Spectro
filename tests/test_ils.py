"""Тесты модуля spectrolib.ils."""

import numpy as np
import pytest

from spectrolib.ils import (
    GaussILS, LorentzILS, VoigtILS, FromFileILS,
    fwhm_to_sigma, sigma_to_fwhm, gauss_convolve,
)


class TestUtils:

    def test_fwhm_sigma_roundtrip(self):
        for fwhm in [0.001, 1.0, 25.0]:
            assert sigma_to_fwhm(fwhm_to_sigma(fwhm)) == pytest.approx(fwhm)

    def test_fwhm_sigma_known(self):
        # FWHM = 2.3548 σ
        assert fwhm_to_sigma(2.3548) == pytest.approx(1.0, rel=1e-3)


class TestGaussILS:

    def test_normalization(self):
        """∫ kernel · dx = 1 (внутренне нормировано через сумму·шаг)."""
        ils = GaussILS(fwhm=1.0)
        ker = ils.kernel(grid_step=0.01)
        # ker уже умножено на шаг → сумма ≈ 1
        assert ker.sum() == pytest.approx(1.0, rel=1e-3)

    def test_convolve_preserves_constant(self):
        """Свёртка с нормированным ядром не меняет константу."""
        grid = np.linspace(0, 100, 10001)
        const = np.full_like(grid, 7.0)
        ils = GaussILS(fwhm=2.0)
        out = ils.convolve(const, grid)
        assert np.allclose(out, 7.0, atol=1e-6)

    def test_convolve_widens_delta(self):
        """Свёртка δ-функции (узкого пика) с гауссом → гаусс той же ширины."""
        grid = np.linspace(-50, 50, 10001)  # шаг 0.01
        delta = np.zeros_like(grid)
        delta[len(grid) // 2] = 1.0
        ils = GaussILS(fwhm=1.0)
        out = ils.convolve(delta, grid)
        # FWHM выходного — оценим через 2·|x при out=max/2|
        peak = out.max()
        above = out >= peak / 2
        idx = np.where(above)[0]
        fwhm_out = grid[idx[-1]] - grid[idx[0]]
        # Ожидаем FWHM ≈ 1.0 (форма ядра + дельта = ядро)
        assert fwhm_out == pytest.approx(1.0, abs=0.05)

    def test_kernel_symmetry(self):
        ils = GaussILS(fwhm=1.0)
        ker = ils.kernel(grid_step=0.01)
        assert np.allclose(ker, ker[::-1], atol=1e-12)


class TestLorentzILS:

    def test_normalization(self):
        ils = LorentzILS(fwhm=1.0)
        ker = ils.kernel(grid_step=0.01)
        # У Лоренца длинные хвосты → нужно широкое окно для точной нормировки
        # Но т.к. мы нормируем через сумму·шаг, должно быть ровно 1
        assert ker.sum() == pytest.approx(1.0, rel=1e-3)

    def test_kernel_symmetry(self):
        ker = LorentzILS(fwhm=1.0).kernel(0.01)
        assert np.allclose(ker, ker[::-1], atol=1e-12)


class TestVoigtILS:

    def test_normalization(self):
        ker = VoigtILS(fwhm_g=1.0, fwhm_l=1.0).kernel(0.01)
        assert ker.sum() == pytest.approx(1.0, rel=1e-3)

    def test_voigt_reduces_to_gauss_when_lorentz_narrow(self):
        """При очень узкой лоренц-компоненте Voigt → Гаусс."""
        grid = np.linspace(-20, 20, 4001)
        delta = np.zeros_like(grid)
        delta[len(grid) // 2] = 1.0
        gauss_out = GaussILS(fwhm=1.0).convolve(delta, grid)
        voigt_out = VoigtILS(fwhm_g=1.0, fwhm_l=1e-5).convolve(delta, grid)
        # На основной части профиля должны совпадать с хорошей точностью
        mid = slice(len(grid) // 2 - 200, len(grid) // 2 + 200)
        assert np.allclose(gauss_out[mid], voigt_out[mid], atol=1e-3)

    def test_voigt_reduces_to_lorentz_when_gauss_narrow(self):
        """При очень узкой гаусс-компоненте Voigt → Лоренц."""
        grid = np.linspace(-50, 50, 10001)
        delta = np.zeros_like(grid)
        delta[len(grid) // 2] = 1.0
        lor_out = LorentzILS(fwhm=1.0).convolve(delta, grid)
        voigt_out = VoigtILS(fwhm_g=1e-4, fwhm_l=1.0).convolve(delta, grid)
        # Сравниваем форму в окрестности пика
        mid = slice(len(grid) // 2 - 1000, len(grid) // 2 + 1000)
        assert np.allclose(lor_out[mid], voigt_out[mid], atol=5e-3)


class TestFromFileILS:

    def test_constructed_from_gaussian_works_like_gaussian(self):
        """Строим FromFileILS из гауссова профиля и сверяем с GaussILS."""
        x = np.linspace(-10, 10, 2001)
        sigma = fwhm_to_sigma(1.0)
        intensity = np.exp(-0.5 * (x / sigma) ** 2)

        from_file = FromFileILS(x, intensity)
        analytic = GaussILS(fwhm=1.0)

        # Эстимат FWHM из дискретных точек должен быть близок к 1.0
        assert from_file.fwhm == pytest.approx(1.0, abs=0.02)

        # Свёртка с обеими ILS должна давать почти одно и то же
        grid = np.linspace(-20, 20, 4001)
        delta = np.zeros_like(grid)
        delta[len(grid) // 2] = 1.0
        out_ff = from_file.convolve(delta, grid)
        out_an = analytic.convolve(delta, grid)
        assert np.allclose(out_ff, out_an, atol=2e-3)

    def test_auto_center(self):
        """Если максимум смещён, auto_center=True перенесёт его в 0."""
        x = np.linspace(0, 20, 401)
        sigma = fwhm_to_sigma(1.0)
        intensity = np.exp(-0.5 * ((x - 10) / sigma) ** 2)
        ils = FromFileILS(x, intensity, auto_center=True)
        # После центрирования профиль должен быть симметричен относительно 0
        assert np.argmax(ils._intensity) == np.argmin(np.abs(ils._offset))

    def test_negative_intensities_clipped(self):
        x = np.linspace(-5, 5, 101)
        intensity = np.exp(-x ** 2) - 0.1   # местами < 0
        ils = FromFileILS(x, intensity, clip_negative=True)
        assert np.all(ils._intensity >= 0)


class TestBackwardCompatibility:

    def test_gauss_convolve_function(self):
        """Старая функция gauss_convolve должна работать как раньше."""
        grid = np.linspace(0, 100, 1001)
        signal = np.zeros_like(grid)
        signal[500] = 1.0
        out_old = gauss_convolve(signal, grid, fwhm=2.0)
        out_new = GaussILS(fwhm=2.0).convolve(signal, grid)
        # Численно близко (точно совпадать не обязаны — реализация чуть разная)
        assert np.allclose(out_old, out_new, atol=1e-3)
