"""
spectrolib.io
=============
Загрузчик спектров из CSV/TXT/XLSX с метаданными в шапке.
"""

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
