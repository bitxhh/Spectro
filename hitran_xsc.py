"""
spectrolib.hitran_xsc
=====================
Загрузчик HITRAN Absorption Cross-Sections (формат `.xsc`).

Зачем
-----
HITRAN xsc — раздел HITRAN с панорамными сечениями поглощения для молекул,
у которых нет/не хватает line-by-line. Сюда вошла, в частности, оригинальная
база PNNL Sharpe et al. (2004) для VOC в ИК-диапазоне (~600–6500 см⁻¹) при
278/298/323 K и атмосферном давлении (broadener N₂). Эти данные используются
библиотекой как источник ИК-сечений ароматики, алифатики, карбонилов
и фоновых спиртов (см. таблицу 2.2/2.4 практической части диплома).

Формат файла
------------
Текстовый, начинается заголовком фиксированной ширины (102 символа):

    formula (a20)  nu_min (f10.4)  nu_max (f10.4)  n_points (i7)
    T_K (f7.2)  P_Torr (f6.1)  sigma_max (e10.3)  resolution (f6.4|a6)
    common_name (a15)  broadener (a4)  ref (i3)

Далее — n_points значений σ [см²/молекула], по 10 в строку, форматом %10.3e.
Сетка ν равномерная: ν_i = ν_min + i·(ν_max − ν_min)/(n_points − 1).

В отличие от родного PNNL (две колонки, A на 1 ppm·m), HITRAN xsc даёт
σ напрямую — конвертация не нужна.

Раскладка
---------
Папка-молекула, внутри — все `.xsc` для разных температур (и, возможно,
для разных спектральных полос одного и того же измерения):

    hitran_xsc_data/
        benzene/
            C6H6_278.0_760.0_600.0-6500.0_09.xsc
            C6H6_298.0_760.0_600.0-6500.0_09.xsc
            C6H6_323.0_760.0_600.0-6500.0_09.xsc
        acetone/
            CH3COCH3_296.0_759.0_2615.0-3300.0_13.xsc    # CH-полосы
            CH3COCH3_297.8_700.0_700.0-1780.0_13.xsc     # CO-полосы

Имя молекулы для GasMixture — имя подпапки.

T-интерполяция
--------------
В каждой точке сетки ν линейно интерполируется σ между двумя ближайшими
доступными температурами, окружающими T_target (по умолчанию 310 K —
температура выдоха). Файлы с близкими T (|ΔT| ≤ snap_T_tol) считаются
относящимися к одной точке по T и при этом могут покрывать разные
спектральные полосы (как у ацетона выше) — тогда σ-спектры
конкатенируются по ν.

Источник данных
---------------
https://hitran.org/xsc/ — раздел Absorption Cross-Sections; фильтр по
Sharpe et al. (2004), broadener N₂, P = 760 Torr.
"""

from __future__ import annotations

import os
import re
import glob as _glob
from typing import Optional

import numpy as np


# Целевая температура выдоха (диплом, аналогично pnnl/mpi).
T_EXHALE_K = 310.0

# Снэп температур: файлы с |ΔT| ≤ этого значения относятся к одной T-точке
# (для случаев типа ацетона: 296.0 + 297.8 K → один T-бин).
DEFAULT_SNAP_T_TOL_K = 2.0

# Мягкий клипп: если T_target отстоит от ближайшей доступной T не более чем
# на это значение, молча зажимаем к крайнему бину (без ошибки). Для VOC в
# ИК-диапазоне ∂lnσ/∂T ≈ 10⁻³/K — 15 K даёт <2% эффект, что меньше прочих
# источников погрешности (например, биологической вариабельности концентраций).
# Нужно, в частности, чтобы ацетон при единственном T = 296.9 K корректно
# подавался на запрос 310 K (выдох).
DEFAULT_SOFT_CLAMP_T_TOL_K = 20.0


# Реестр зарегистрированных файлов:
#   _HXSC_TABLES[name] = list of dicts {'T_K': float, 'nu_min': float,
#                                       'nu_max': float, 'path': str,
#                                       'header': dict}
_HXSC_TABLES: dict = {}
_DB_PATH: Optional[str] = None


# ---------------------------------------------------------------------------
# Парсер
# ---------------------------------------------------------------------------

def parse_xsc_header(line: str) -> dict:
    """
    Разобрать заголовок HITRAN xsc.

    Формат — фиксированной ширины, но реализации иногда подмешивают
    разделители-пробелы. Используем split с устойчивостью к лишним пробелам
    и проверкой числа полей.

    Returns
    -------
    dict
        Ключи: formula, nu_min, nu_max, n_points, T_K, P_Torr, sigma_max,
        resolution (float|None), common_name, broadener, ref.
    """
    parts = line.split()
    if len(parts) < 8:
        raise ValueError(f"HITRAN xsc: слишком короткий заголовок: {line!r}")

    formula = parts[0]
    nu_min = float(parts[1])
    nu_max = float(parts[2])
    n_points = int(parts[3])
    T_K = float(parts[4])
    P_Torr = float(parts[5])

    # sigma_max: может «слипнуться» с resolution из-за формата %.3E%6.4f
    # без пробела. Учитываем оба варианта.
    raw_sig_res = parts[6]
    m = re.match(r'^([-+]?\d+\.?\d*[Ee][-+]?\d+)(.*)$', raw_sig_res)
    if m and m.group(2):
        sigma_max = float(m.group(1))
        res_str = m.group(2)
        rest = parts[7:]
    else:
        sigma_max = float(raw_sig_res)
        res_str = parts[7] if len(parts) > 7 else ''
        rest = parts[8:]

    try:
        resolution = float(res_str) if res_str else None
    except ValueError:
        resolution = None

    # Оставшиеся поля: common_name (может содержать пробелы), broadener, ref.
    # Стандартный порядок в xsc: name, broadener, ref. Берём с конца.
    ref = None
    broadener = None
    common_name = None
    if rest:
        # ref — последнее число
        try:
            ref = int(rest[-1])
            rest = rest[:-1]
        except (ValueError, IndexError):
            pass
        if rest:
            broadener = rest[-1]
            rest = rest[:-1]
        if rest:
            common_name = ' '.join(rest)

    return {
        'formula': formula,
        'nu_min': nu_min,
        'nu_max': nu_max,
        'n_points': n_points,
        'T_K': T_K,
        'P_Torr': P_Torr,
        'sigma_max': sigma_max,
        'resolution': resolution,
        'common_name': common_name,
        'broadener': broadener,
        'ref': ref,
    }


def parse_xsc_file(path: str):
    """
    Прочитать HITRAN xsc-файл.

    Returns
    -------
    nu_cm : np.ndarray
        Равномерная сетка волновых чисел, см⁻¹, по возрастанию.
    sigma_cm2 : np.ndarray
        Сечение поглощения, см²/молекула.
    header : dict
        Распарсенный заголовок.
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        header_line = fh.readline()
        if not header_line:
            raise ValueError(f"HITRAN xsc: пустой файл: {path}")
        header = parse_xsc_header(header_line)

        # Все оставшиеся значения — σ.
        values = []
        for line in fh:
            for tok in line.split():
                try:
                    values.append(float(tok))
                except ValueError:
                    continue

    n_expected = header['n_points']
    sigma = np.asarray(values[:n_expected], dtype=float)
    if sigma.size < n_expected:
        raise ValueError(
            f"HITRAN xsc: в {path} найдено {sigma.size} значений σ, "
            f"ожидалось {n_expected}"
        )

    nu_cm = np.linspace(header['nu_min'], header['nu_max'], n_expected)
    return nu_cm, sigma, header


# ---------------------------------------------------------------------------
# Инициализация и регистрация
# ---------------------------------------------------------------------------

def init_db(path: str = 'hitran_xsc_data'):
    """
    Проиндексировать локальную папку с xsc-файлами.

    Раскладка: подпапка на молекулу (имя подпапки = имя молекулы в GasMixture).
    Допускается также плоская раскладка — тогда имя берётся из header.formula
    или из имени файла до первого '_'.

    Файлы только индексируются (читается лишь заголовок); полные σ-массивы
    читаются лениво в load_xsc_sigma.
    """
    global _DB_PATH
    _DB_PATH = path
    os.makedirs(path, exist_ok=True)

    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            for fp in sorted(_glob.glob(os.path.join(full, '*.xsc'))):
                _index_file(entry, fp)
        elif os.path.isfile(full) and full.lower().endswith('.xsc'):
            # Плоская раскладка — имя из header.formula либо имени файла.
            _index_file(None, full)


def _index_file(name: Optional[str], path: str):
    """Прочитать только заголовок и зарегистрировать запись."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            header = parse_xsc_header(fh.readline())
    except Exception:
        return  # битый файл — пропускаем тихо, как в pnnl/mpi

    if name is None:
        # Имя по умолчанию — formula из заголовка.
        name = header.get('formula') or os.path.splitext(os.path.basename(path))[0]

    rec = {
        'T_K': header['T_K'],
        'nu_min': header['nu_min'],
        'nu_max': header['nu_max'],
        'path': path,
        'header': header,
    }
    bucket = _HXSC_TABLES.setdefault(name, [])
    if all(r['path'] != path for r in bucket):
        bucket.append(rec)


def register_xsc_file(name: str, path: str):
    """
    Явно зарегистрировать xsc-файл молекулы — путь, не зависящий от
    соглашений об именах папок.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"HITRAN xsc: файл не найден: {path}")
    _index_file(name, path)


def list_local_molecules() -> list:
    """Список молекул, для которых найдены xsc-файлы."""
    return sorted(_HXSC_TABLES)


def list_files(name: str) -> list:
    """Метаданные всех зарегистрированных файлов молекулы."""
    return list(_HXSC_TABLES.get(name, []))


def available_temperatures(name: str, snap_T_tol: float = DEFAULT_SNAP_T_TOL_K):
    """Список доступных T (после снэпа близких файлов) для молекулы."""
    if name not in _HXSC_TABLES:
        return []
    return sorted(_group_by_T(_HXSC_TABLES[name], snap_T_tol))


# ---------------------------------------------------------------------------
# Загрузка σ с T-интерполяцией
# ---------------------------------------------------------------------------

def _group_by_T(records: list, snap_T_tol: float) -> dict:
    """
    Сгруппировать файлы по близким температурам.

    Returns
    -------
    dict {T_repr: [records]}, где T_repr — средняя T в группе.
    """
    if not records:
        return {}
    sorted_recs = sorted(records, key=lambda r: r['T_K'])
    groups = []
    cur = [sorted_recs[0]]
    for r in sorted_recs[1:]:
        if r['T_K'] - cur[-1]['T_K'] <= snap_T_tol:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    return {float(np.mean([r['T_K'] for r in g])): g for g in groups}


def _concat_band_spectra(records: list):
    """
    Прочитать все σ-спектры из списка записей и склеить их по ν.

    Полосы либо не пересекаются (как у ацетона: 700–1780 + 2615–3300),
    либо совпадают (один и тот же файл повторно). В точках пересечения
    берётся среднее (на практике перекрытий быть не должно).
    """
    pieces = []
    for r in records:
        nu, sig, _ = parse_xsc_file(r['path'])
        order = np.argsort(nu)
        pieces.append((nu[order], sig[order]))

    if len(pieces) == 1:
        return pieces[0]

    # Сшиваем по ν. Если полосы не перекрываются — просто конкатенация
    # отсортированных кусков.
    pieces.sort(key=lambda p: p[0][0])
    nu_all = np.concatenate([p[0] for p in pieces])
    sig_all = np.concatenate([p[1] for p in pieces])
    order = np.argsort(nu_all)
    nu_all = nu_all[order]
    sig_all = sig_all[order]

    # Уникализация совпадающих узлов (усреднение σ).
    nu_u, inv = np.unique(nu_all, return_inverse=True)
    if nu_u.size == nu_all.size:
        return nu_u, sig_all
    sig_u = np.zeros_like(nu_u)
    counts = np.zeros_like(nu_u)
    np.add.at(sig_u, inv, sig_all)
    np.add.at(counts, inv, 1.0)
    return nu_u, sig_u / counts


def load_xsc_sigma(name: str, T_target: float = T_EXHALE_K,
                   snap_T_tol: float = DEFAULT_SNAP_T_TOL_K,
                   soft_clamp_T_tol: float = DEFAULT_SOFT_CLAMP_T_TOL_K,
                   allow_extrapolation: bool = False,
                   verbose: bool = False):
    """
    Вернуть σ(ν) молекулы при T_target, линейно интерполируя по T в каждой
    точке сетки. Аналог spectrolib.pnnl.load_pnnl_sigma, но для HITRAN-xsc.

    Алгоритм:
      1) Если есть T-бин, совпадающий с T_target (|ΔT| ≤ snap_T_tol) —
         читаем все его файлы (полосы конкатенируются по ν) и возвращаем σ
         без интерполяции по T.
      2) Иначе берём два T-бина, окружающие T_target; в каждом
         конкатенируем полосы по ν; на общей сетке ν линейно интерполируем
         по T:
             σ(T) = σ_lo + (σ_hi − σ_lo)·(T − T_lo)/(T_hi − T_lo).
      3) Если T_target вне диапазона — ValueError; allow_extrapolation=True
         зажимает к крайнему доступному бину с предупреждением.

    Parameters
    ----------
    name : str
        Имя молекулы (см. init_db / register_xsc_file).
    T_target : float
        Целевая температура, K (по умолчанию 310, как требует диплом).
    snap_T_tol : float
        Допуск близости по T для объединения файлов в один T-бин.
    allow_extrapolation : bool
        Разрешить зажим к крайней T при выходе за диапазон.
    verbose : bool

    Returns
    -------
    nu_cm : np.ndarray
    sigma_cm2 : np.ndarray
    meta : dict
        {'name', 'T_target_K', 'T_used', 'mode'}.
    """
    if name not in _HXSC_TABLES or not _HXSC_TABLES[name]:
        raise KeyError(
            f"HITRAN xsc: для '{name}' нет данных. Сначала "
            f"init_db('hitran_xsc_data') или register_xsc_file(...)."
        )

    groups = _group_by_T(_HXSC_TABLES[name], snap_T_tol)
    temps = sorted(groups)

    # 1) Точное (с учётом snap) совпадение.
    for T in temps:
        if abs(T - T_target) <= snap_T_tol:
            nu, sig = _concat_band_spectra(groups[T])
            if verbose:
                print(f"[hitran_xsc] {name}: T-бин {T:.2f} K "
                      f"({len(groups[T])} файл(ов))")
            return nu, sig, {'name': name, 'T_target_K': T_target,
                              'T_used': (T,), 'mode': 'exact'}

    T_lo_avail, T_hi_avail = temps[0], temps[-1]

    # 3) Вне диапазона.
    if T_target < T_lo_avail or T_target > T_hi_avail:
        T_edge = T_lo_avail if T_target < T_lo_avail else T_hi_avail
        delta = abs(T_target - T_edge)
        if delta <= soft_clamp_T_tol:
            # Мягкий клипп: эффект на σ в ИК пренебрежим, молча зажимаем.
            nu, sig = _concat_band_spectra(groups[T_edge])
            if verbose:
                print(f"[hitran_xsc] {name}: T_target={T_target} → soft-clamp "
                      f"к {T_edge:.2f} K (ΔT={delta:.1f} K ≤ "
                      f"{soft_clamp_T_tol} K)")
            return nu, sig, {'name': name, 'T_target_K': T_target,
                              'T_used': (T_edge,), 'mode': 'soft_clamp'}
        if not allow_extrapolation:
            raise ValueError(
                f"HITRAN xsc: T_target={T_target} K вне диапазона "
                f"[{T_lo_avail:.1f}, {T_hi_avail:.1f}] K для '{name}' "
                f"(ΔT={delta:.1f} K > {soft_clamp_T_tol} K мягкого допуска). "
                f"Передайте allow_extrapolation=True или добавьте файл, "
                f"окружающий {T_target} K."
            )
        nu, sig = _concat_band_spectra(groups[T_edge])
        if verbose:
            print(f"[hitran_xsc] {name}: T_target={T_target} вне диапазона → "
                  f"hard-clamp к {T_edge:.2f} K")
        return nu, sig, {'name': name, 'T_target_K': T_target,
                          'T_used': (T_edge,), 'mode': 'clamped'}

    # 2) Bracket.
    idx = np.searchsorted(temps, T_target)
    T_hi = temps[idx]
    T_lo = temps[idx - 1]
    nu_lo, sig_lo = _concat_band_spectra(groups[T_lo])
    nu_hi, sig_hi = _concat_band_spectra(groups[T_hi])

    # Общая сетка ν — объединение узлов, ограниченное пересечением диапазонов
    # (за пересечением «честно» интерполировать нельзя).
    nu_min = max(nu_lo[0], nu_hi[0])
    nu_max = min(nu_lo[-1], nu_hi[-1])
    if nu_max <= nu_min:
        raise ValueError(
            f"HITRAN xsc: спектры {name} при {T_lo} и {T_hi} K не имеют "
            f"общего ν-диапазона; T-интерполяция невозможна."
        )
    nu_common = np.union1d(nu_lo, nu_hi)
    nu_common = nu_common[(nu_common >= nu_min) & (nu_common <= nu_max)]

    sig_lo_i = np.interp(nu_common, nu_lo, sig_lo)
    sig_hi_i = np.interp(nu_common, nu_hi, sig_hi)

    w = (T_target - T_lo) / (T_hi - T_lo)
    sigma_T = sig_lo_i + (sig_hi_i - sig_lo_i) * w

    if verbose:
        print(f"[hitran_xsc] {name}: интерполяция по T {T_lo:.2f}→{T_hi:.2f} "
              f"на {T_target} K (w={w:.3f}), узлов: {nu_common.size}")

    return nu_common, sigma_T, {'name': name, 'T_target_K': T_target,
                                 'T_used': (T_lo, T_hi), 'mode': 'interp'}
