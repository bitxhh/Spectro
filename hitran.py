"""
spectrolib.hitran
=================
Обёртки над hapi для работы с базой HITRAN.

Молекулы (M, I) для основных биомаркеров — справочник:
    H2O: (1, 1)
    CO2: (2, 1)
    O3:  (3, 1)
    N2O: (4, 1)
    CO:  (5, 1)
    CH4: (6, 1)
    O2:  (7, 1)
    NO:  (8, 1)
    NO2: (10, 1)
    NH3: (11, 1)
"""

import os
import shutil

import hapi


# Опорная температура HITRAN: сила линий S(ν₀) в базе табулирована именно
# при этой T. hapi.absorptionCoefficient_* при Environment={'T': T_K,...}
# автоматически пересчитывает S(T) через функцию распределения Q(T)/Q(T_ref)
# и больцмановский фактор exp[-c2·E"·(1/T - 1/T_ref)]. То есть «температурную
# коррекцию сечений» делает hapi, наша работа — передать ему T_K корректно.
T_REF_HITRAN_K = 296.0


# Справочник: имя молекулы → (molecule_id, isotopologue_id)
MOLECULE_IDS = {
    'H2O':   (1, 1),
    'CO2':   (2, 1),
    'O3':    (3, 1),
    'N2O':   (4, 1),
    'CO':    (5, 1),
    'CH4':   (6, 1),
    'O2':    (7, 1),
    'NO':    (8, 1),
    'NO2':   (10, 1),
    'NH3':   (11, 1),
    'HCHO':  (20, 1),   # формальдегид (H2CO) — ИК через HITRAN line list
    'CH3OH': (39, 1),   # метанол — ИК через HITRAN line list
}


_DB_INITIALIZED = False


def init_db(path='hitran_data'):
    """
    Инициализирует локальную БД HITRAN. Создаёт папку если её нет.
    Вызывается автоматически при первом обращении к fetch_molecule.
    """
    global _DB_INITIALIZED
    if not _DB_INITIALIZED:
        os.makedirs(path, exist_ok=True)
        hapi.db_begin(path)
        _DB_INITIALIZED = True


def fetch_molecule(name, table_name=None, nu_min=None, nu_max=None,
                   wl_min_nm=None, wl_max_nm=None,
                   force=False):
    """
    Скачивает данные о линиях молекулы из HITRAN.

    Можно задавать диапазон в волновых числах (nu_min/nu_max, см⁻¹)
    или в нанометрах (wl_min_nm/wl_max_nm) — функция выберет правильно.

    Parameters
    ----------
    name : str
        Имя молекулы из MOLECULE_IDS (например 'O2', 'CO2').
    table_name : str, optional
        Имя локальной таблицы. По умолчанию: name + диапазон.
    nu_min, nu_max : float, optional
        Диапазон волновых чисел, см⁻¹.
    wl_min_nm, wl_max_nm : float, optional
        Диапазон длин волн, нм.
    force : bool
        Если True — перекачать даже если таблица уже есть.

    Returns
    -------
    str
        Имя созданной таблицы (для передачи в Spectrum.add_molecule).

    Examples
    --------
    >>> fetch_molecule('O2', wl_min_nm=759, wl_max_nm=775)
    'O2_759-775nm'
    """
    init_db()

    if name not in MOLECULE_IDS:
        raise ValueError(
            f"Молекула '{name}' не найдена. Доступны: {list(MOLECULE_IDS)}"
        )
    M, I = MOLECULE_IDS[name]

    # Определяем диапазон в см⁻¹
    if nu_min is None and nu_max is None:
        if wl_min_nm is None or wl_max_nm is None:
            raise ValueError(
                "Задай либо (nu_min, nu_max), либо (wl_min_nm, wl_max_nm)"
            )
        # нм → см⁻¹: больший λ даёт меньший ν
        nu_min = 1e7 / wl_max_nm
        nu_max = 1e7 / wl_min_nm

    # Имя таблицы по умолчанию
    if table_name is None:
        if wl_min_nm is not None:
            table_name = f"{name}_{wl_min_nm:.0f}-{wl_max_nm:.0f}nm"
        else:
            table_name = f"{name}_{nu_min:.0f}-{nu_max:.0f}cm"

    # Проверяем что таблица уже скачана
    if not force and table_name in hapi.tableList():
        return table_name

    hapi.fetch(table_name, M, I, nu_min, nu_max)
    return table_name


def list_local_tables():
    """Возвращает список локально доступных таблиц HITRAN."""
    init_db()
    return list(hapi.tableList())


def clear_cache(table=None, db_path='hitran_data'):
    """
    Удалить локальные таблицы HITRAN.

    Parameters
    ----------
    table : str | None
        Если задано — удалить только эту таблицу (файлы `.data` + `.header`
        и in-memory запись в hapi.LOCAL_TABLE_CACHE).
        Если None — снести весь кеш (всю папку db_path и сбросить in-memory).
    db_path : str
        Путь к папке кеша. По умолчанию 'hitran_data' (как в init_db).

    Returns
    -------
    list[str]
        Имена удалённых таблиц.

    Examples
    --------
    >>> clear_cache('O2_759-775nm')           # одна таблица
    >>> clear_cache()                         # весь кеш
    """
    global _DB_INITIALIZED
    cache = getattr(hapi, 'LOCAL_TABLE_CACHE', None)
    removed = []

    if table is not None:
        for ext in ('.data', '.header'):
            p = os.path.join(db_path, table + ext)
            if os.path.exists(p):
                os.remove(p)
        if cache is not None and table in cache:
            del cache[table]
        removed.append(table)
        return removed

    # Полная очистка
    if os.path.exists(db_path):
        for fname in os.listdir(db_path):
            if fname.endswith('.data'):
                removed.append(fname[:-len('.data')])
        shutil.rmtree(db_path)
    if cache is not None:
        cache.clear()
    _DB_INITIALIZED = False
    return removed
