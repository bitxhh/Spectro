"""Тесты модуля panels: загрузка, генерация, преконцентрирование."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spectrolib import (
    MixturePanel, Biomarker, load_mixture_panel,
    Instrument, NoiseModel, SpectrumGenerator, GaussILS,
)


# ---------------------------------------------------------------------------
# Biomarker
# ---------------------------------------------------------------------------

class TestBiomarker:

    def test_from_dict_with_ppm(self):
        b = Biomarker.from_dict({'name': 'CO2', 'c_ppm': 40000})
        assert b.name == 'CO2'
        assert b.c_ppm == 40000
        assert b.c_ppb == 40000 * 1000

    def test_from_dict_with_ppb(self):
        b = Biomarker.from_dict({'name': 'NO', 'c_ppb': 25})
        assert b.c_ppm == 0.025
        assert b.c_ppb == 25

    def test_from_dict_no_concentration_raises(self):
        with pytest.raises(ValueError, match='c_ppm или c_ppb'):
            Biomarker.from_dict({'name': 'CO2'})

    def test_from_dict_both_concentrations_raises(self):
        with pytest.raises(ValueError, match='одновременно'):
            Biomarker.from_dict({'name': 'CO2', 'c_ppm': 1, 'c_ppb': 1})

    def test_optional_fields(self):
        b = Biomarker.from_dict({
            'name': 'CO',
            'c_ppm': 1.5,
            'wavelength_nm': 4666,
            'source': 'Risby 2006',
            'notes': 'курильщики выше',
        })
        assert b.wavelength_nm == 4666
        assert b.source == 'Risby 2006'
        assert b.notes == 'курильщики выше'


# ---------------------------------------------------------------------------
# MixturePanel — базовое
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_panel_dict():
    return {
        'name': 'Test panel',
        'reference': 'Test reference 2024',
        'notes': 'demo',
        'conditions': {'T_K': 310, 'p_atm': 1.0, 'L_cm': 100},
        'biomarkers': [
            {'name': 'CO', 'c_ppm': 1.5, 'source': 'src1'},
            {'name': 'NH3', 'c_ppb': 500},
        ],
    }


@pytest.fixture
def simple_panel(simple_panel_dict):
    return MixturePanel.from_dict(simple_panel_dict)


class TestPanelBasic:

    def test_from_dict(self, simple_panel):
        assert simple_panel.name == 'Test panel'
        assert len(simple_panel) == 2
        assert simple_panel.biomarkers[0].name == 'CO'
        assert simple_panel.biomarkers[1].c_ppm == 0.5

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match='name'):
            MixturePanel.from_dict({'biomarkers': [{'name': 'CO', 'c_ppm': 1}]})

    def test_no_biomarkers_raises(self):
        with pytest.raises(ValueError, match='биомаркеры'):
            MixturePanel.from_dict({'name': 'Empty', 'biomarkers': []})

    def test_iteration(self, simple_panel):
        names = [b.name for b in simple_panel]
        assert names == ['CO', 'NH3']

    def test_repr(self, simple_panel):
        r = repr(simple_panel)
        assert 'Test panel' in r
        assert '2 biomarkers' in r


class TestPanelToMixture:

    def test_to_mixture_basic(self, simple_panel):
        mix = simple_panel.to_mixture()
        assert mix.composition == {'CO': 1.5, 'NH3': 0.5}
        assert mix.T_K == 310
        assert mix.p_atm == 1.0
        assert mix.L_cm == 100

    def test_to_mixture_uses_defaults(self):
        # Панель без conditions — должны быть дефолты
        panel = MixturePanel.from_dict({
            'name': 'P',
            'biomarkers': [{'name': 'CO', 'c_ppm': 1}],
        })
        mix = panel.to_mixture()
        assert mix.T_K == 310.0
        assert mix.p_atm == 1.0
        assert mix.L_cm == 10.0


class TestPanelScaled:

    def test_scaled_multiplies_concentrations(self, simple_panel):
        scaled = simple_panel.scaled(100)
        assert scaled.biomarkers[0].c_ppm == 1.5 * 100
        assert scaled.biomarkers[1].c_ppm == 0.5 * 100

    def test_scaled_preserves_other_fields(self, simple_panel):
        scaled = simple_panel.scaled(100)
        assert scaled.name == simple_panel.name
        assert scaled.reference == simple_panel.reference
        assert scaled.conditions == simple_panel.conditions

    def test_scaled_is_copy(self, simple_panel):
        scaled = simple_panel.scaled(100)
        assert simple_panel.biomarkers[0].c_ppm == 1.5  # оригинал не тронут
        assert scaled.biomarkers[0].c_ppm == 150

    def test_scaled_factor_one_no_op(self, simple_panel):
        scaled = simple_panel.scaled(1.0)
        # Должна быть та же концентрация, без пометки [scaled by 1.0]
        assert scaled.biomarkers[0].c_ppm == 1.5
        assert '[scaled' not in (scaled.notes or '')


# ---------------------------------------------------------------------------
# Сериализация
# ---------------------------------------------------------------------------

class TestPanelIO:

    def test_save_load_yaml_roundtrip(self, simple_panel):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'panel.yaml'
            simple_panel.save(path)
            loaded = MixturePanel.from_file(path)

        assert loaded.name == simple_panel.name
        assert len(loaded) == len(simple_panel)
        assert loaded.biomarkers[0].name == simple_panel.biomarkers[0].name
        assert loaded.biomarkers[0].c_ppm == simple_panel.biomarkers[0].c_ppm

    def test_save_load_json_roundtrip(self, simple_panel):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'panel.json'
            simple_panel.save(path)
            loaded = MixturePanel.from_file(path)

        assert loaded.name == simple_panel.name
        assert loaded.conditions == simple_panel.conditions

    def test_unknown_extension_raises(self, simple_panel):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'panel.txt'
            with pytest.raises(ValueError, match='yaml'):
                simple_panel.save(path)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            MixturePanel.from_file('/nonexistent/panel.yaml')

    def test_load_mixture_panel_shortcut(self, simple_panel):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'panel.yaml'
            simple_panel.save(path)
            loaded = load_mixture_panel(path)
        assert loaded.name == simple_panel.name


# ---------------------------------------------------------------------------
# Сводка
# ---------------------------------------------------------------------------

class TestPanelSummary:

    def test_text_summary(self, simple_panel):
        s = simple_panel.summary(fmt='text')
        assert 'Test panel' in s
        assert 'CO' in s
        assert 'NH3' in s
        # Концентрации показаны в ppb
        assert '1500' in s   # 1.5 ppm = 1500 ppb

    def test_markdown_summary(self, simple_panel):
        s = simple_panel.summary(fmt='markdown')
        assert s.startswith('# Test panel')
        assert '|' in s   # есть таблица
        assert 'Test reference 2024' in s


# ---------------------------------------------------------------------------
# Пример-панель из репозитория
# ---------------------------------------------------------------------------

class TestExamplePanels:

    def test_breath_demo_loads(self):
        # Пример лежит в spectrolib/example_panels/
        import spectrolib
        pkg_dir = Path(spectrolib.__file__).parent
        panel_path = pkg_dir / 'example_panels' / 'breath_demo.yaml'
        if not panel_path.exists():
            pytest.skip(f"Example panel not found: {panel_path}")

        panel = load_mixture_panel(panel_path)
        assert panel.name
        assert len(panel) > 0
        # Все молекулы должны быть в MOLECULE_IDS
        from spectrolib import MOLECULE_IDS
        for b in panel:
            assert b.name in MOLECULE_IDS, (
                f"Молекула {b.name} из примера breath_demo.yaml "
                f"отсутствует в MOLECULE_IDS"
            )

    def test_lung_cancer_template_loads(self):
        import spectrolib
        pkg_dir = Path(spectrolib.__file__).parent
        panel_path = pkg_dir / 'example_panels' / 'lung_cancer_template.yaml'
        if not panel_path.exists():
            pytest.skip()
        # Этот шаблон должен загружаться, но молекулы могут отсутствовать
        # в MOLECULE_IDS (комментарий в самом файле объясняет).
        panel = load_mixture_panel(panel_path)
        assert 'cancer' in panel.name.lower()


# ---------------------------------------------------------------------------
# Интеграция с генератором (без HITRAN, через _StructuredGenerator)
# ---------------------------------------------------------------------------

class _SyntheticGenerator(SpectrumGenerator):
    """
    Тестовый генератор: не использует HITRAN. Целиком переопределяет
    generate(), а не _generate_clean, потому что generate() в базовом
    классе не идёт через _generate_clean.
    """
    def generate(self, mixture):
        spec = self.instrument.empty_spectrum()
        # Каждая молекула — гауссов пик (заменяет HITRAN-расчёт)
        for i, (name, c_ppm) in enumerate(mixture.composition.items()):
            spec.add_gauss_peak(
                center_nm=755 + i * 3,
                fwhm_nm=0.5,
                amplitude=c_ppm * 1e-6,
            )
        if self.instrument.ils is not None:
            spec.convolve_ils(self.instrument.ils)
        # Заполнение molecules вручную, потому что add_gauss_peak их не пишет
        spec.molecules = [
            {'name': name, 'c_ppm': c_ppm, 'L_cm': mixture.L_cm,
             'T_K': mixture.T_K, 'p_atm': mixture.p_atm,
             'profile': 'voigt', 'table_name': 'demo',
             'diluent': mixture.diluent or {'air': 1.0}}
            for name, c_ppm in mixture.composition.items()
        ]
        if self.noise_model is not None:
            seed = (None if self.seed is None
                    else self.seed + self._seed_offset)
            spec.add_noise_model(self.noise_model, seed=seed)
            self._seed_offset += 1
        return spec


class TestPanelGenerate:

    def test_generate_one_command(self, simple_panel):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.05)
        gen = _SyntheticGenerator(instrument=inst)
        spec = simple_panel.generate(gen)
        # Имя панели в meta
        assert spec.meta['panel_name'] == 'Test panel'
        assert spec.meta['panel_reference'] == 'Test reference 2024'
        assert spec.meta['panel_n_biomarkers'] == 2
        # Молекулы из панели — в спектре
        names = [m['name'] for m in spec.molecules]
        assert 'CO' in names
        assert 'NH3' in names

    def test_plot_uses_panel_name_as_title(self, simple_panel):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.05)
        gen = _SyntheticGenerator(instrument=inst)
        spec = simple_panel.generate(gen)
        fig, ax = spec.plot()
        title = ax.get_title()
        assert 'Test panel' in title
        plt.close(fig)
