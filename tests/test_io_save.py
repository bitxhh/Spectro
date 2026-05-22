"""Тесты сохранения спектров (CSV / NPZ) и roundtrip через load_spectrum."""

import json
import os

import numpy as np
import pytest

from spectrolib import Spectrum, load_spectrum, save_spectrum


@pytest.fixture
def simple_spec():
    return (Spectrum.from_range(750, 760, step_nm=0.01)
            .add_gauss_peak(755, fwhm_nm=0.5, amplitude=0.3)
            .add_noise(sigma=0.005, seed=0))


class TestSaveCSV:

    def test_creates_file(self, simple_spec, tmp_path):
        out = tmp_path / 'spec.csv'
        result = save_spectrum(simple_spec, out)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_method_alias(self, simple_spec, tmp_path):
        out = tmp_path / 'spec.csv'
        simple_spec.save(out)
        assert out.exists()

    def test_header_contains_metadata(self, simple_spec, tmp_path):
        out = tmp_path / 'spec.csv'
        simple_spec.save(out)
        text = out.read_text(encoding='utf-8')
        assert 'spectrolib_version' in text
        assert 'kind' in text
        assert 'metadata_json' in text

    def test_roundtrip_via_load_spectrum(self, simple_spec, tmp_path):
        out = tmp_path / 'spec.csv'
        simple_spec.save(out, include_true=False)  # одна колонка для load_spectrum
        wl, vals, meta = load_spectrum(str(out))
        assert len(wl) == len(simple_spec.wavelength_nm)
        np.testing.assert_allclose(wl, simple_spec.wavelength_nm, rtol=1e-5)
        np.testing.assert_allclose(vals, simple_spec.transmittance, rtol=1e-5)
        assert meta.get('kind') == 'transmittance'

    def test_kind_absorbance(self, simple_spec, tmp_path):
        out = tmp_path / 'spec_a.csv'
        simple_spec.save(out, kind='absorbance', include_true=False)
        wl, vals, meta = load_spectrum(str(out))
        np.testing.assert_allclose(vals, simple_spec.absorbance, rtol=1e-5)
        assert meta['kind'] == 'absorbance'


class TestSaveNPZ:

    def test_creates_file(self, simple_spec, tmp_path):
        out = tmp_path / 'spec.npz'
        simple_spec.save(out)
        assert out.exists()

    def test_contains_all_arrays(self, simple_spec, tmp_path):
        out = tmp_path / 'spec.npz'
        simple_spec.save(out)
        archive = np.load(out, allow_pickle=False)
        for key in ('wavelength_nm', 'observed', 'true',
                    'clean_optical_depth', 'noise_T', 'noise_OD'):
            assert key in archive.files
        np.testing.assert_allclose(archive['wavelength_nm'], simple_spec.wavelength_nm)
        np.testing.assert_allclose(archive['observed'], simple_spec.transmittance)
        np.testing.assert_allclose(archive['true'], simple_spec.true_transmittance)

    def test_metadata_json_is_parseable(self, simple_spec, tmp_path):
        out = tmp_path / 'spec.npz'
        simple_spec.save(out)
        archive = np.load(out, allow_pickle=False)
        meta = json.loads(str(archive['metadata_json']))
        assert meta['kind'] == 'transmittance'
        assert meta['n_points'] == len(simple_spec.wavelength_nm)


class TestSaveErrors:

    def test_unknown_format_raises(self, simple_spec, tmp_path):
        with pytest.raises(ValueError):
            simple_spec.save(tmp_path / 'spec.xyz')

    def test_invalid_kind_raises(self, simple_spec, tmp_path):
        with pytest.raises(ValueError):
            simple_spec.save(tmp_path / 'spec.csv', kind='garbage')

    def test_explicit_fmt(self, simple_spec, tmp_path):
        out = tmp_path / 'noextension'
        simple_spec.save(out, fmt='csv')
        assert out.exists()
