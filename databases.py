"""
spectrolib.databases
=====================
Реестр источников сечений и разрешение источника для молекулы.

В библиотеке три источника сечений поглощения σ [см²/молекула]:

    'hitran' — line-by-line через hapi (ИК/ближний ИК, основные газы);
    'pnnl'   — PNNL IR Database (ИК-сечения VOC, которых нет в линейном
               списке HITRAN; T-интерполяция до 310 K, см. spectrolib.pnnl);
    'mpi'    — MPI-Mainz UV/VIS Spectral Atlas (УФ-сечения, см. spectrolib.mpi).

Все три приводятся к ОДНОЙ величине — σ [см²/молекула] на сетке волновых
чисел (см⁻¹), после чего Spectrum.add_molecule считает OD одинаково
(закон Бугера–Ламберта–Бера). Это и есть «согласованность баз»: единица,
сетка и формула перехода σ→OD общие; различается только способ получить σ.

Разрешение источника (порядок приоритета):
    1) явный аргумент source= в add_molecule / sources= в GasMixture;
    2) запись в реестре MOLECULE_SOURCE (через register_molecule);
    3) дефолт DB_HITRAN.
"""

DB_HITRAN = 'hitran'
DB_PNNL = 'pnnl'
DB_MPI = 'mpi'
DB_HITRAN_XSC = 'hitran_xsc'

VALID_SOURCES = (DB_HITRAN, DB_PNNL, DB_MPI, DB_HITRAN_XSC)


# Реестр «молекула → предпочтительный источник».
# Намеренно небольшой и явный: сюда попадают молекулы, для которых
# HITRAN-линейный список НЕ является источником по умолчанию.
# Имена — те же, что используются в загрузчиках pnnl/mpi (имя файла/папки).
#
# Список можно расширять во время работы через register_molecule().
MOLECULE_SOURCE = {
    # --- VOC-биомаркеры рака лёгких, ИК-полосы → HITRAN xsc (Sharpe/PNNL) ---
    # Таблицы 2.2 и 2.4 практической части. Файлы лежат в hitran_xsc_data/<name>/.
    'benzene':      DB_HITRAN_XSC,   # C6H6
    'toluene':      DB_HITRAN_XSC,   # C7H8
    'ethylbenzene': DB_HITRAN_XSC,   # C8H10
    'styrene':      DB_HITRAN_XSC,   # C8H8
    'acetone':      DB_HITRAN_XSC,   # C3H6O
    'butanone':     DB_HITRAN_XSC,   # C4H8O — 2-бутанон
    'hexanal':      DB_HITRAN_XSC,   # C6H12O
    'pentane':      DB_HITRAN_XSC,   # C5H12
    'hexane':       DB_HITRAN_XSC,   # C6H14
    'isoprene':     DB_HITRAN_XSC,   # C5H8
    'ethanol':      DB_HITRAN_XSC,   # C2H5OH

    # --- УФ-полосы тех же VOC → MPI-Mainz Spectral Atlas ---
    # Те же подпапки в mpi_data/ — обращение по тем же именам, но с
    # явным source='mpi' в add_molecule (или sources=).
    # ИК-полоса метанола берётся через HITRAN line list (молекула 39).
    'HCHO':         DB_MPI,          # формальдегид, УФ 280–360 нм
    'SO2':          DB_MPI,          # УФ ~200-320 нм
    'glyoxal':      DB_MPI,          # CHOCHO, УФ-ВИД ~210-470 нм
}


def register_molecule(name, source):
    """
    Зарегистрировать предпочтительный источник для молекулы.

    Parameters
    ----------
    name : str
        Имя молекулы (как в загрузчике соответствующей базы).
    source : {'hitran', 'pnnl', 'mpi'}
    """
    if source not in VALID_SOURCES:
        raise ValueError(
            f"Источник '{source}' неизвестен. Доступны: {VALID_SOURCES}"
        )
    MOLECULE_SOURCE[name] = source


def resolve_source(name, source=None, overrides=None):
    """
    Определить источник сечений для молекулы.

    Parameters
    ----------
    name : str
        Имя молекулы.
    source : str, optional
        Явный источник. Если задан — имеет высший приоритет.
    overrides : dict, optional
        Локальная карта {имя: источник} (например, sources= из GasMixture),
        приоритетнее глобального реестра, но ниже явного source.

    Returns
    -------
    str
        Один из VALID_SOURCES.
    """
    if source is not None:
        if source not in VALID_SOURCES:
            raise ValueError(
                f"Источник '{source}' неизвестен. Доступны: {VALID_SOURCES}"
            )
        return source
    if overrides and name in overrides:
        src = overrides[name]
        if src not in VALID_SOURCES:
            raise ValueError(
                f"Источник '{src}' для '{name}' неизвестен. "
                f"Доступны: {VALID_SOURCES}"
            )
        return src
    return MOLECULE_SOURCE.get(name, DB_HITRAN)
