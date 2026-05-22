"""
spectrolib.io
=============
Загрузка/сохранение спектров (CSV/TXT/XLSX/NPZ) с метаданными в шапке.
"""

import json
import os

import numpy as np
import pandas as pd


def load_spectrum(filepath):
    """
    Загружает спектр из файла и возвращает (wavelength, values, metadata).

    Поддерживаемые форматы: CSV, TXT, XLSX.
    Автоматически определяет разделитель и десятичный знак.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    wavelength : np.ndarray
    values : np.ndarray
    metadata : dict
        Из шапки файла (строки с '#' в формате 'ключ: значение').

        Заполняется **только** для текстовых форматов (csv/txt).
        Для xlsx метаданные не парсятся и `metadata` всегда возвращается
        пустым словарём ({}); храни сопутствующие данные отдельно
        или конвертируй файл в csv с '#'-шапкой.

    Notes
    -----
    Подводный камень: русский Excel экспортирует с ';' и ',' как
    десятичным знаком. Функция перебирает варианты автоматически.
    После загрузки всегда смотри dtypes — если object, парсинг сломан.
    """
    filepath = str(filepath)
    ext = filepath.rsplit('.', 1)[-1].lower()
    metadata = {}

    if ext in ('csv', 'txt'):
        # Парсим шапку: # ключ: значение
        with open(filepath, encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    content = line.lstrip('#').strip()
                    if ':' in content:
                        key, _, val = content.partition(':')
                        metadata[key.strip().lower()] = val.strip()
                else:
                    break

        # Перебираем разделитель × десятичный знак
        candidates = []
        for sep in (';', ',', '\t', r'\s+'):
            for decimal in (',', '.'):
                try:
                    df = pd.read_csv(
                        filepath, comment='#', sep=sep, decimal=decimal,
                        header=None,
                        engine='python' if sep == r'\s+' else 'c',
                    )
                    df = df.apply(pd.to_numeric, errors='coerce')
                    df = df.dropna(axis=1, how='all').dropna()
                    if df.shape[1] >= 2 and len(df) > 5:
                        candidates.append(df)
                except Exception:
                    continue

        if not candidates:
            raise ValueError(f"Не удалось распарсить: {filepath}")

        # Выбор лучшего варианта парсинга:
        # 1) предпочитаем те, где первая колонка монотонна (это сетка λ или ν)
        # 2) среди них — самый длинный
        def _is_monotonic(s):
            x = s.iloc[:, 0].values
            return bool(np.all(np.diff(x) > 0)) or bool(np.all(np.diff(x) < 0))

        monotonic_candidates = [c for c in candidates if _is_monotonic(c)]
        if monotonic_candidates:
            df = max(monotonic_candidates, key=len)
        else:
            # Fallback: тот же max-length, но с warning
            import warnings as _w
            _w.warn(
                f"Ни один вариант парсинга {filepath} не дал монотонную "
                "первую колонку — берём самый длинный. Проверь результат."
            )
            df = max(candidates, key=len)

    elif ext == 'xlsx':
        df = pd.read_excel(filepath, header=None)
        # Пропускаем нечисловую шапку
        for i, row in df.iterrows():
            try:
                float(row.iloc[0])
                df = df.iloc[i:].reset_index(drop=True)
                break
            except (ValueError, TypeError):
                continue
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

    else:
        raise ValueError(
            f"Неизвестное расширение: {ext}. Поддержка: csv, txt, xlsx."
        )

    return df.iloc[:, 0].values, df.iloc[:, 1].values, metadata


# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------

def _values_for(spec, kind, which):
    if which == 'observed':
        return {
            'transmittance': spec.transmittance,
            'absorbance':    spec.absorbance,
            'optical_depth': spec.optical_depth,
        }[kind]
    return {
        'transmittance': spec.true_transmittance,
        'absorbance':    spec.true_absorbance,
        'optical_depth': spec.true_optical_depth,
    }[kind]


def _build_metadata(spec, kind):
    """Собрать сериализуемый dict с описанием спектра."""
    return {
        'kind':       kind,
        'n_points':   int(len(spec.wavelength_nm)),
        'wl_min_nm':  float(spec.wavelength_nm.min()),
        'wl_max_nm':  float(spec.wavelength_nm.max()),
        'molecules':  list(spec.molecules),
        'ils':        spec.ils,
        'noise':      spec.noise,
        'history':    list(spec.history),
        'meta':       dict(spec.meta),
    }


def save_spectrum(spec, filepath, kind='transmittance', fmt=None,
                  include_true=True):
    """
    Сохранить спектр в файл.

    Parameters
    ----------
    spec : Spectrum
    filepath : str | os.PathLike
    kind : {'transmittance', 'absorbance', 'optical_depth'}
        Какую величину писать как «наблюдаемую» колонку.
    fmt : {'csv', 'npz'} | None
        Если None — определяется по расширению.

        - 'csv' — текст с шапкой `# ключ: значение`. Совместим с load_spectrum.
        - 'npz' — бинарный архив со всеми массивами (wavelength, observed,
          true, _clean_optical_depth, _noise_T, _noise_OD) и метаданными
          в JSON. Полный архив для долгого хранения.
    include_true : bool
        Для CSV: добавить колонку true_<kind> рядом с наблюдаемой.

    Returns
    -------
    str
        Абсолютный путь к сохранённому файлу.

    Examples
    --------
    >>> spec.save('out.csv')                # CSV с метаданными в шапке
    >>> spec.save('out.npz')                # полный архив для архивации
    >>> spec.save('out.csv', kind='absorbance')
    """
    if kind not in ('transmittance', 'absorbance', 'optical_depth'):
        raise ValueError(f"kind должен быть transmittance/absorbance/optical_depth, получено {kind!r}")

    filepath = os.fspath(filepath)
    if fmt is None:
        ext = filepath.rsplit('.', 1)[-1].lower() if '.' in filepath else ''
        if ext in ('csv', 'txt'):
            fmt = 'csv'
        elif ext == 'npz':
            fmt = 'npz'
        else:
            raise ValueError(
                f"Не могу определить формат по расширению '{ext}'. Укажи fmt= явно."
            )

    metadata = _build_metadata(spec, kind)
    observed = _values_for(spec, kind, 'observed')

    if fmt == 'csv':
        _save_csv(filepath, spec.wavelength_nm, observed,
                  true_values=_values_for(spec, kind, 'true') if include_true else None,
                  metadata=metadata, kind=kind)
    elif fmt == 'npz':
        np.savez(
            filepath,
            wavelength_nm=spec.wavelength_nm,
            observed=observed,
            true=_values_for(spec, kind, 'true'),
            clean_optical_depth=spec._clean_optical_depth,
            noise_T=spec._noise_T,
            noise_OD=spec._noise_OD,
            metadata_json=json.dumps(metadata, ensure_ascii=False, default=str),
        )
    else:
        raise ValueError(f"fmt должен быть 'csv' или 'npz', получено {fmt!r}")

    return os.path.abspath(filepath)


def _save_csv(filepath, wavelength, observed, *, true_values, metadata, kind):
    """CSV с шапкой `# ключ: значение`, два-три столбца значений."""
    from . import __version__ as _v
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# spectrolib_version: {_v}\n")
        f.write(f"# kind: {kind}\n")
        f.write(f"# n_points: {metadata['n_points']}\n")
        f.write(f"# wl_min_nm: {metadata['wl_min_nm']:.6f}\n")
        f.write(f"# wl_max_nm: {metadata['wl_max_nm']:.6f}\n")
        if metadata['molecules']:
            mols = '; '.join(
                f"{m.get('name', '?')}({m.get('c_ppm', '?')} ppm)"
                for m in metadata['molecules']
            )
            f.write(f"# molecules: {mols}\n")
            m0 = metadata['molecules'][0]
            for k in ('T_K', 'p_atm', 'L_cm'):
                if k in m0:
                    f.write(f"# {k}: {m0[k]}\n")
        if metadata['ils']:
            f.write(f"# ils: {metadata['ils']}\n")
        if metadata['noise']:
            f.write(f"# noise: {metadata['noise']}\n")
        # Полный JSON в одну строку — для машинного восстановления, если нужно
        f.write(f"# metadata_json: {json.dumps(metadata, ensure_ascii=False, default=str)}\n")

        # Заголовок колонок
        if true_values is not None:
            f.write(f"# columns: wavelength_nm,{kind},true_{kind}\n")
            for wl, obs, tru in zip(wavelength, observed, true_values):
                f.write(f"{wl:.6f},{obs:.8e},{tru:.8e}\n")
        else:
            f.write(f"# columns: wavelength_nm,{kind}\n")
            for wl, obs in zip(wavelength, observed):
                f.write(f"{wl:.6f},{obs:.8e}\n")
