"""Тесты на clear_cache — без обращения к серверу HITRAN."""

import os

import pytest

import hapi
from spectrolib import hitran as h
from spectrolib.hitran import clear_cache


def _make_fake_table(db_path, name):
    """Создать пару файлов .data + .header, имитирующих кешированную таблицу."""
    os.makedirs(db_path, exist_ok=True)
    for ext in ('.data', '.header'):
        with open(os.path.join(db_path, name + ext), 'w') as f:
            f.write('# fake\n')


class TestClearCacheSingleTable:

    def test_removes_files(self, tmp_path):
        db = str(tmp_path / 'hitran_cache')
        _make_fake_table(db, 'O2_test')
        removed = clear_cache('O2_test', db_path=db)
        assert removed == ['O2_test']
        assert not os.path.exists(os.path.join(db, 'O2_test.data'))
        assert not os.path.exists(os.path.join(db, 'O2_test.header'))

    def test_removes_from_in_memory_cache(self, tmp_path):
        db = str(tmp_path / 'hitran_cache')
        _make_fake_table(db, 'CO2_test')
        # Имитация in-memory записи
        hapi.LOCAL_TABLE_CACHE['CO2_test'] = {'fake': True}
        clear_cache('CO2_test', db_path=db)
        assert 'CO2_test' not in hapi.LOCAL_TABLE_CACHE

    def test_missing_table_is_noop(self, tmp_path):
        # Не должно падать, если файлов нет
        db = str(tmp_path / 'hitran_cache')
        result = clear_cache('not_there', db_path=db)
        assert result == ['not_there']


class TestClearCacheAll:

    def test_removes_whole_dir(self, tmp_path):
        db = str(tmp_path / 'hitran_cache')
        _make_fake_table(db, 'A')
        _make_fake_table(db, 'B')
        removed = clear_cache(db_path=db)
        assert set(removed) == {'A', 'B'}
        assert not os.path.exists(db)

    def test_resets_initialized_flag(self, tmp_path):
        db = str(tmp_path / 'hitran_cache')
        _make_fake_table(db, 'X')
        h._DB_INITIALIZED = True
        clear_cache(db_path=db)
        assert h._DB_INITIALIZED is False

    def test_missing_dir_is_noop(self, tmp_path):
        db = str(tmp_path / 'nonexistent')
        # Не должно упасть
        result = clear_cache(db_path=db)
        assert result == []
