"""Тесты объектного фасада (без сетевых вызовов HITRAN)."""

import numpy as np
import pytest

from spectrolib import (
    Instrument, GasMixture, NoiseModel, SpectrumGenerator,
    GaussILS, Spectrum, MixturePanel, preconcentrate,
)


class TestInstrument:

    def test_empty_spectrum(self):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.01)
        s = inst.empty_spectrum()
        assert isinstance(s, Spectrum)
        assert s.wavelength_nm[0] == pytest.approx(750)
        assert s.wavelength_nm[-1] == pytest.approx(770, abs=0.01)


class TestSpectrumGeneratorWithoutHITRAN:
    """
    Тесты работают без HITRAN — используем пустую смесь и проверяем,
    что пайплайн (ILS, шум, метаданные) собирается корректно.
    """

    def test_empty_mixture(self):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.01)
        gen = SpectrumGenerator(inst)
        # Пустая композиция — допустима, спектр будет нулевой
        spec = gen.generate(GasMixture(composition={}))
        assert np.allclose(spec.true_optical_depth, 0)

    def test_ils_applied(self):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.001,
                          ils=GaussILS(fwhm=0.5))
        gen = SpectrumGenerator(inst)
        spec = gen.generate(GasMixture(composition={}))
        assert spec.ils is not None
        assert spec.ils['type'] == 'GaussILS'

    def test_noise_applied(self):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.01)
        noise = NoiseModel(thermal_sigma=0.01)
        gen = SpectrumGenerator(inst, noise_model=noise, seed=42)
        spec = gen.generate(GasMixture(composition={}))
        assert spec.noise is not None
        # Истина — нули, наблюдаемое — с шумом
        assert np.allclose(spec.true_optical_depth, 0)
        assert not np.allclose(spec.optical_depth, 0)

    def test_seed_increment_between_calls(self):
        """Два последовательных вызова generate с одним seed должны давать
        разные шумовые реализации (внутренний счётчик увеличивается)."""
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.01)
        noise = NoiseModel(thermal_sigma=0.01)
        gen = SpectrumGenerator(inst, noise_model=noise, seed=42)
        s1 = gen.generate(GasMixture(composition={}))
        s2 = gen.generate(GasMixture(composition={}))
        # Разные реализации шума
        assert not np.allclose(s1.optical_depth, s2.optical_depth)

    def test_meta_populated(self):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.01,
                          ils=GaussILS(fwhm=0.5))
        gen = SpectrumGenerator(inst, seed=7)
        mix = GasMixture(composition={}, T_K=310, p_atm=1.0, L_cm=10.0)
        spec = gen.generate(mix)
        assert 'instrument' in spec.meta
        assert 'mixture' in spec.meta
        assert spec.meta['mixture']['T_K'] == 310
        assert spec.meta['generator_seed'] == 7


class TestGasMixtureWithT:

    def test_with_T_changes_only_T(self):
        mix = GasMixture(composition={'CO': 1.5, 'NO': 0.025},
                         T_K=296.0, p_atm=1.0, L_cm=10.0,
                         diluent={'air': 1.0}, profile='voigt')
        new = mix.with_T(310.0)
        assert new.T_K == 310.0
        assert new.composition == mix.composition
        assert new.p_atm == mix.p_atm
        assert new.L_cm == mix.L_cm
        assert new.diluent == mix.diluent
        assert new.profile == mix.profile

    def test_with_T_is_copy(self):
        mix = GasMixture(composition={'CO': 1.5}, T_K=296.0)
        new = mix.with_T(310.0)
        # Оригинал не тронут
        assert mix.T_K == 296.0
        # composition — независимый dict
        new.composition['CO'] = 999
        assert mix.composition['CO'] == 1.5


class TestGasMixturePreconcentrated:

    def test_multiplies_concentrations(self):
        mix = GasMixture(composition={'CO': 1.5, 'NO': 0.025})
        new = mix.preconcentrated(1000)
        assert new.composition == {'CO': 1500.0, 'NO': 25.0}

    def test_preserves_other_fields(self):
        mix = GasMixture(composition={'CO': 1.5}, T_K=310.0,
                         p_atm=1.0, L_cm=10.0,
                         diluent={'air': 0.96, 'self': 0.04},
                         profile='lorentz')
        new = mix.preconcentrated(100)
        assert new.T_K == mix.T_K
        assert new.p_atm == mix.p_atm
        assert new.L_cm == mix.L_cm
        assert new.diluent == mix.diluent
        assert new.profile == mix.profile

    def test_is_copy(self):
        mix = GasMixture(composition={'CO': 1.5})
        new = mix.preconcentrated(100)
        assert mix.composition == {'CO': 1.5}     # оригинал не тронут
        assert new.composition == {'CO': 150.0}

    def test_invalid_K_pre_raises(self):
        mix = GasMixture(composition={'CO': 1.5})
        with pytest.raises(ValueError, match='K_pre'):
            mix.preconcentrated(0)
        with pytest.raises(ValueError, match='K_pre'):
            mix.preconcentrated(-1)

    def test_chain_with_with_L(self):
        mix = GasMixture(composition={'NO': 0.025}, L_cm=10.0)
        new = mix.preconcentrated(1000).with_L(100.0)
        assert new.composition == {'NO': 25.0}
        assert new.L_cm == 100.0


class TestPreconcentrateDispatch:

    def test_dispatches_on_gas_mixture(self):
        mix = GasMixture(composition={'NO': 0.025})
        out = preconcentrate(mix, 1000)
        assert isinstance(out, GasMixture)
        assert out.composition == {'NO': 25.0}

    def test_dispatches_on_mixture_panel(self):
        panel = MixturePanel.from_dict({
            'name': 'P',
            'biomarkers': [{'name': 'NO', 'c_ppb': 25}],
        })
        out = preconcentrate(panel, 1000)
        assert isinstance(out, MixturePanel)
        # 25 ppb × 1000 = 25 ppm
        assert out.biomarkers[0].c_ppm == 25.0

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match='GasMixture или MixturePanel'):
            preconcentrate({'NO': 25}, 1000)
        with pytest.raises(TypeError):
            preconcentrate(42, 1000)
