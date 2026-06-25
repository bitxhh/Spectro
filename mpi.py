"""
spectrolib.mpi
==============
Загрузчик УФ/ВИД-сечений из MPI-Mainz UV/VIS Spectral Atlas
(Max-Planck-Institut für Chemie, Mainz).

Зачем
-----
Атлас MPI-Mainz — основной источник сечений поглощения в УФ-диапазоне для
молекул вроде O3, NO2, SO2, HCHO, глиоксаля. Для биомаркеров/газового
анализа в УФ это дополняет HITRAN (у которого в УФ — раздел absorption
cross-sections, с ним и сверяемся на перекрытии).

Формат файла
------------
Текст, две колонки:
    col1 — длина волны λ [нм] (в ВОЗДУХЕ),
    col2 — сечение σ [см²/молекула].
Иногда есть 3-я колонка — предел погрешности (игнорируем).
Строки-заголовки/комментарии (первая колонка — не число) пропускаются.

Имя файла кодирует метаданные:
    formula_author(year)_temperature_wavelengths.txt
например:
    CHOCHO_Horowitz(2001)_295K_210-470nm(int-c).txt
    NO2_Vandaele(1998)_294K_238-1000nm.txt

Длина волны: воздух vs вакуум
-----------------------------
λ в атласе даны для воздуха; HITRAN оперирует вакуумными волновыми числами.
Сдвиг ~0.03 % — пренебрежимо мал на фоне допусков сверки (10-20 %), поэтому
по умолчанию переводим λ_air[нм] → ν[см⁻¹] напрямую как 1e7/λ. При желании
можно включить поправку воздух→вакуум (air_to_vacuum=True).

Источник данных
---------------
http://satellite.mpic.de/spectral_atlas (он же uv-vis-spectral-atlas-mainz.org);
объёмная выгрузка — на Zenodo. Файлы скачиваются пользователем вручную
(программный доступ к Zenodo блокируется), парсер работает с локальными
файлами в папке mpi_data/.
"""

import os
import re
import glob as _glob

import numpy as np

from .physics import nm_to_wavenumber


# Целевая температура выдоха (для информативности; в УФ σ почти не зависит
# от T в окрестности комнатной, поэтому здесь — выбор ближайшего файла по T,
# а не интерполяция, если не задано иное).
T_EXHALE_K = 310.0


# Реестр: name → list[dict(meta + path)]
_MPI_TABLES = {}
_DB_PATH = None


# Стандартный показатель преломления воздуха (Edlén/Ciddor, упрощённо) —
# для опциональной поправки air→vacuum. λ_vac = λ_air · n_air.
def _n_air(wavelength_nm):
    """Показатель преломления воздуха (приближение Edlén) для λ в нм."""
    # σ² в мкм⁻²
    inv_um2 = (1e3 / np.asarray(wavelength_nm, dtype=float)) ** 2
    n_minus_1 = 1e-8 * (
        8342.54 + 2406147.0 / (130.0 - inv_um2) + 15998.0 / (38.9 - inv_um2)
    )
    return 1.0 + n_minus_1


# ---------------------------------------------------------------------------
# Парсинг
# ---------------------------------------------------------------------------

def parse_mpi_file(path):
    """
    Прочитать MPI-файл → (lambda_nm, sigma_cm2) в порядке файла.

    Заголовки/комментарии пропускаются; 3-я колонка (погрешность) — игнор.

    Returns
    -------
    lambda_nm : np.ndarray
    sigma_cm2 : np.ndarray
    """
    lams = []
    sigs = []
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('//') or s.startswith('%'):
                continue
            parts = re.split(r'[\s,;]+', s)
            if len(parts) < 2:
                continue
            try:
                lam = float(parts[0])
                sig = float(parts[1])
            except ValueError:
                continue
            lams.append(lam)
            sigs.append(sig)

    if not lams:
        raise ValueError(f"В файле MPI не найдено числовых строк: {path}")

    return np.asarray(lams, dtype=float), np.asarray(sigs, dtype=float)


def parse_mpi_filename(path):
    """
    Разобрать имя файла MPI-Mainz на метаданные.

    Формат: formula_author(year)_temperature_wavelengths.txt

    Returns
    -------
    dict
        Ключи: formula, author, year (int|None), T_K (float|None),
        wl_range (tuple|None), filename. Поля, которые не удалось
        распознать, равны None.
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]

    meta = {
        'formula': None, 'author': None, 'year': None,
        'T_K': None, 'wl_range': None, 'filename': base,
    }

    # формула — до первого '_'
    m = re.match(r'^([A-Za-z0-9]+)_', stem)
    if m:
        meta['formula'] = m.group(1)

    # автор(год)
    m = re.search(r'_([A-Za-z\.\-]+)\((\d{4})\)', stem)
    if m:
        meta['author'] = m.group(1)
        meta['year'] = int(m.group(2))

    # температура: число + K
    m = re.search(r'(?<![0-9])(\d{2,3})\s*[Kk](?![A-Za-z])', stem)
    if m:
        meta['T_K'] = float(m.group(1))

    # диапазон длин волн: a-b nm
    m = re.search(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*nm', stem)
    if m:
        meta['wl_range'] = (float(m.group(1)), float(m.group(2)))

    return meta


# ---------------------------------------------------------------------------
# Инициализация / регистрация
# ---------------------------------------------------------------------------

def init_db(path='mpi_data'):
    """
    Инициализировать локальную папку MPI и проиндексировать файлы.

    Раскладка — любая из:
      * подпапка на молекулу (mpi_data/NO2/NO2_Vandaele(1998)_294K_...txt);
      * плоско (mpi_data/NO2_Vandaele(1998)_294K_...txt).
    Имя молекулы берётся из имени подпапки (если есть) либо из поля formula
    в имени файла.

    Файлы только индексируются (пути + метаданные), читаются лениво.
    """
    global _DB_PATH
    _DB_PATH = path
    os.makedirs(path, exist_ok=True)

    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            for fpath in sorted(_glob.glob(os.path.join(full, '*'))):
                if os.path.isfile(fpath) and _is_data_file(fpath):
                    _index_file(entry, fpath)
        elif os.path.isfile(full) and _is_data_file(full):
            meta = parse_mpi_filename(full)
            name = meta['formula'] or os.path.splitext(entry)[0]
            _index_file(name, full)


def _is_data_file(fpath):
    return os.path.splitext(fpath)[1].lower() in ('.txt', '.csv', '.dat', '.tsv')


def _index_file(name, fpath):
    meta = parse_mpi_filename(fpath)
    meta['path'] = fpath
    _MPI_TABLES.setdefault(name, [])
    # не дублируем один и тот же путь
    if all(rec['path'] != fpath for rec in _MPI_TABLES[name]):
        _MPI_TABLES[name].append(meta)


def register_mpi_file(name, path, T_K=None):
    """
    Явно зарегистрировать MPI-файл молекулы.

    Parameters
    ----------
    name : str
        Имя молекулы (как в составе смеси).
    path : str
        Путь к текстовому файлу (λ[нм], σ[см²]).
    T_K : float, optional
        Температура файла; если None — пытаемся взять из имени.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"MPI: файл не найден: {path}")
    meta = parse_mpi_filename(path)
    meta['path'] = path
    if T_K is not None:
        meta['T_K'] = float(T_K)
    _MPI_TABLES.setdefault(name, [])
    if all(rec['path'] != path for rec in _MPI_TABLES[name]):
        _MPI_TABLES[name].append(meta)


def list_local_molecules():
    """Список молекул, для которых найдены MPI-файлы."""
    return sorted(_MPI_TABLES)


def list_files(name):
    """Метаданные всех зарегистрированных файлов молекулы."""
    return list(_MPI_TABLES.get(name, []))


# ---------------------------------------------------------------------------
# Загрузка σ
# ---------------------------------------------------------------------------

def _select_record(name, T_target):
    """
    Выбрать запись (файл) молекулы, ближайшую по температуре к T_target.
    Если ни у одной нет T — берём первую.
    """
    recs = _MPI_TABLES[name]
    with_T = [r for r in recs if r.get('T_K') is not None]
    if not with_T:
        return recs[0]
    return min(with_T, key=lambda r: abs(r['T_K'] - T_target))


def load_mpi_sigma(name, T_target=T_EXHALE_K, air_to_vacuum=False,
                   verbose=False):
    """
    Вернуть σ(ν) молекулы из MPI-Mainz на сетке волновых чисел [см⁻¹].

    В УФ зависимость σ от T в окрестности комнатной слабая и не всегда
    обеспечена несколькими файлами, поэтому по умолчанию выбирается файл с
    температурой, ближайшей к T_target (без интерполяции по T). При наличии
    данных при нескольких T интерполяцию можно добавить так же, как в pnnl.

    Parameters
    ----------
    name : str
        Имя молекулы (см. init_db / register_mpi_file).
    T_target : float
        Целевая температура, K (для выбора файла). По умолчанию 310 K.
    air_to_vacuum : bool
        Перевести λ воздух→вакуум перед расчётом ν (поправка ~0.03 %).
    verbose : bool
        Диагностика.

    Returns
    -------
    nu_cm : np.ndarray
        Волновые числа, см⁻¹, по возрастанию.
    sigma_cm2 : np.ndarray
        Сечение поглощения, см²/молекула.
    meta : dict
        Метаданные выбранного файла + 'T_target_K'.
    """
    if name not in _MPI_TABLES or not _MPI_TABLES[name]:
        raise KeyError(
            f"MPI: для '{name}' нет данных. Сначала init_db('mpi_data') "
            f"или register_mpi_file('{name}', path)."
        )

    rec = _select_record(name, T_target)
    lam_nm, sigma = parse_mpi_file(rec['path'])

    if air_to_vacuum:
        lam_nm = lam_nm * _n_air(lam_nm)

    nu_cm = nm_to_wavenumber(lam_nm)   # 1e7/λ; порядок переворачивается

    # Упорядочим по возрастанию ν.
    order = np.argsort(nu_cm)
    nu_cm = nu_cm[order]
    sigma = sigma[order]

    out_meta = dict(rec)
    out_meta['T_target_K'] = T_target
    if verbose:
        print(f"[mpi] {name}: файл {rec.get('filename')} "
              f"(T={rec.get('T_K')} K), точек: {nu_cm.size}, "
              f"air_to_vacuum={air_to_vacuum}")

    return nu_cm, sigma, out_meta


def clear_cache():
    """Сбросить in-memory индекс MPI-таблиц (файлы на диске не трогаем)."""
    _MPI_TABLES.clear()
