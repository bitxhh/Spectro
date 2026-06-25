"""
spectrolib.pnnl
===============
Загрузчик ИК-сечений из PNNL IR Database (Pacific Northwest National Lab).

Зачем
-----
PNNL содержит количественные ИК-сечения паров органики (VOC), которых нет
в линейном (line-by-line) списке HITRAN: ацетон, изопрен, метанол, толуол и
т.д. Для биомаркеров выдоха это основной источник в ИК-диапазоне.

Что особенного по физике
-------------------------
PNNL меряет спектры при ТРЁХ температурах: 5, 25 и 50 °C
(= 278.15, 298.15, 323.15 K). Температура выдоха — 310 K — лежит МЕЖДУ
298.15 и 323.15 K, поэтому сечение на 310 K получаем ЛИНЕЙНОЙ интерполяцией
по температуре В КАЖДОЙ ТОЧКЕ СЕТКИ (интерполяция, а не экстраполяция).
Это и есть требование диплома: «линейная интерполяция по T в каждой точке
сетки строго до 310 K».

Единицы PNNL и переход к σ [см²/молекула]
-----------------------------------------
PNNL-спектры нормированы на 1 ppm·m при 1 атм и 296 K, величина —
десятичная поглощательная способность (base-10 absorbance, A = −log10 T).

Закон Бера в наших единицах: OD_e = σ · N · ℓ  (натуральный логарифм),
где OD_e = −ln(T) = A · ln(10).

Колоночная плотность для нормировки PNNL (1 ppm · 1 m при 296 K, 1 атм):
    N·ℓ = X · N_total(296,1) · ℓ,   X = 1e-6,  ℓ = 100 см.
Отсюда
    σ [см²/молекула] = A · ln(10) / (X · N_total(296,1) · ℓ).
Численно N_total(296,1)·100·1e-6 ≈ 2.48e15 молекул/см² ⇒ σ ≈ A · 9.3e-16.

Важно: σ — фундаментальная характеристика молекулы и НЕ зависит от того,
при какой T мы потом считаем OD. Поэтому перевод A→σ всегда использует
опорные условия нормировки PNNL (296 K, 1 атм). Температурная зависимость
ФОРМЫ спектра передаётся отбором/интерполяцией исходных файлов по T, а
числовая плотность газа N(T,p) в законе Бера считается отдельно в
Spectrum.add_molecule по запрошенной T_K.

Формат файла
------------
Простой текст, две колонки: волновое число [см⁻¹] и поглощательная
способность (на 1 ppm·m). Строки-заголовки/комментарии (не начинаются с
числа) пропускаются. Разделитель — пробелы/табы/запятые.

Источник данных
---------------
Полный многотемпературный набор: PNNL IR Database (Sharpe et al. 2004),
доступ через Globus/ORCID. Имена температур в файлах распознаются по
маркерам _5C_/_25C_/_50C_ или 278K/298K/323K. Можно также передавать
явную карту files={T_K: path}.
"""

import os
import re
import glob as _glob

import numpy as np

from .physics import number_density


# Опорные условия нормировки PNNL.
T_REF_PNNL_K = 296.0
P_REF_PNNL_ATM = 1.0
# Нормировка PNNL: 1 ppm по объёму на 1 метр пути.
PNNL_PPM = 1e-6          # мольная доля, соответствующая 1 ppm
PNNL_PATH_CM = 100.0     # 1 метр в см

# Целевая температура выдоха (диплом): интерполяция строго сюда.
T_EXHALE_K = 310.0

_LN10 = np.log(10.0)


# Маркеры температуры в имени файла → T в кельвинах.
# Цельсий: 5/25/50 °C (стандартные точки PNNL). Кельвин: прямое значение.
_T_CELSIUS_MARKERS = {
    5: 278.15,
    25: 298.15,
    50: 323.15,
}


# Реестр загруженных молекул: name → {T_K: (nu_cm_asc, sigma_cm2)}
# Заполняется register_pnnl_files / автодискавери в init_db.
_PNNL_TABLES = {}
_DB_PATH = None


# ---------------------------------------------------------------------------
# Парсинг файла и имени
# ---------------------------------------------------------------------------

def parse_pnnl_file(path):
    """
    Прочитать PNNL-файл → (nu_cm, absorbance), обе колонки в порядке файла.

    Возвращает волновое число [см⁻¹] и поглощательную способность
    (base-10, на 1 ppm·m). Заголовки и комментарии пропускаются.

    Returns
    -------
    nu_cm : np.ndarray
    absorbance : np.ndarray
    """
    nus = []
    vals = []
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('//'):
                continue
            # Разделители: запятая/точка-с-запятой/пробелы.
            parts = re.split(r'[\s,;]+', s)
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                # строка-заголовок (не число в первой колонке) — пропускаем
                continue
            nus.append(x)
            vals.append(y)

    if not nus:
        raise ValueError(f"В файле PNNL не найдено числовых строк: {path}")

    nu_cm = np.asarray(nus, dtype=float)
    absorbance = np.asarray(vals, dtype=float)
    return nu_cm, absorbance


def parse_pnnl_temperature(path):
    """
    Извлечь температуру [K] из имени файла PNNL.

    Распознаёт маркеры _5C_/_25C_/_50C_ (°C по стандартным точкам PNNL)
    и явные кельвины вида 278K / 298K / 323K / 310K.

    Returns
    -------
    float | None
        Температура в K либо None, если не удалось распознать.
    """
    base = os.path.basename(path)

    # Явные кельвины: число + 'K' (не часть большего слова).
    m = re.search(r'(?<![0-9])(\d{3})\s*[Kk](?![A-Za-z])', base)
    if m:
        return float(m.group(1))

    # Цельсий: число + 'C' (например 25C, 5C, 50C, _25C_).
    m = re.search(r'(?<![0-9])(\d{1,2})\s*[Cc](?![A-Za-z])', base)
    if m:
        c = int(m.group(1))
        if c in _T_CELSIUS_MARKERS:
            return _T_CELSIUS_MARKERS[c]
        return c + 273.15

    return None


# ---------------------------------------------------------------------------
# Перевод поглощательной способности PNNL → σ
# ---------------------------------------------------------------------------

def pnnl_absorbance_to_sigma(absorbance):
    """
    Перевести поглощательную способность PNNL (base-10, на 1 ppm·m)
    в сечение поглощения σ [см²/молекула].

        σ = A · ln(10) / (X · N_total(296,1) · ℓ),
        X = 1 ppm = 1e-6,  ℓ = 100 см.

    Колоночная плотность считается через physics.number_density при
    опорных условиях PNNL (296 K, 1 атм) — для единообразия с остальной
    библиотекой.
    """
    column_density = (
        PNNL_PPM * number_density(T_REF_PNNL_K, P_REF_PNNL_ATM) * PNNL_PATH_CM
    )  # молекул/см²
    return np.asarray(absorbance, dtype=float) * _LN10 / column_density


# ---------------------------------------------------------------------------
# Инициализация / регистрация данных
# ---------------------------------------------------------------------------

def init_db(path='pnnl_data'):
    """
    Инициализировать локальную папку PNNL и провести автодискавери файлов.

    Ожидаемая раскладка: подпапка на молекулу, внутри — текстовые файлы по
    температурам, например:

        pnnl_data/
            acetone/
                acetone_5C.txt
                acetone_25C.txt
                acetone_50C.txt
            isoprene/
                isoprene_278K.txt
                ...

    Допускается также «плоская» раскладка: файлы вида <name>_<T>.txt прямо
    в корне папки. Температура берётся из имени (parse_pnnl_temperature).

    Файлы НЕ читаются здесь целиком — только индексируются пути; чтение
    и перевод в σ происходят лениво в load_pnnl_sigma.
    """
    global _DB_PATH
    _DB_PATH = path
    os.makedirs(path, exist_ok=True)

    # Папки-молекулы.
    for entry in sorted(os.listdir(path)):
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            files = _discover_files_in_dir(full)
            if files:
                _PNNL_TABLES.setdefault(entry, {}).update(files)

    # Плоская раскладка: <name>_<T>.txt в корне.
    for fpath in _glob.glob(os.path.join(path, '*.txt')):
        T = parse_pnnl_temperature(fpath)
        if T is None:
            continue
        name = _name_from_flat_file(fpath)
        if name:
            _PNNL_TABLES.setdefault(name, {})[T] = fpath


def _discover_files_in_dir(dir_path):
    """Собрать {T_K: path} из текстовых файлов одной папки-молекулы."""
    found = {}
    for fpath in sorted(_glob.glob(os.path.join(dir_path, '*'))):
        if not os.path.isfile(fpath):
            continue
        if os.path.splitext(fpath)[1].lower() not in ('.txt', '.csv', '.dat', '.tsv'):
            continue
        T = parse_pnnl_temperature(fpath)
        if T is not None:
            found[T] = fpath
    return found


def _name_from_flat_file(fpath):
    """Имя молекулы из плоского файла <name>_<T>.<ext> → '<name>'."""
    base = os.path.splitext(os.path.basename(fpath))[0]
    # отрезаем хвост с температурой
    base = re.sub(r'[_-]?\d{1,3}\s*[KkCc](?![A-Za-z]).*$', '', base)
    return base or None


def register_pnnl_files(name, files):
    """
    Явно зарегистрировать файлы молекулы: надёжный путь, не зависящий от
    соглашений об именах.

    Parameters
    ----------
    name : str
        Имя молекулы (как в составе смеси / реестре источников).
    files : dict
        Карта {T_K: path}. Минимум один элемент. Для интерполяции до 310 K
        нужны точки, окружающие 310 (т.е. как минимум 298.15 и 323.15 K).
    """
    norm = {}
    for T, p in files.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"PNNL: файл не найден: {p}")
        norm[float(T)] = p
    if not norm:
        raise ValueError("register_pnnl_files: пустая карта files")
    _PNNL_TABLES.setdefault(name, {}).update(norm)


def list_local_molecules():
    """Список молекул, для которых найдены PNNL-файлы."""
    return sorted(_PNNL_TABLES)


def available_temperatures(name):
    """Отсортированный список доступных температур [K] для молекулы."""
    if name not in _PNNL_TABLES:
        return []
    return sorted(_PNNL_TABLES[name])


# ---------------------------------------------------------------------------
# Загрузка σ с интерполяцией по T
# ---------------------------------------------------------------------------

def _load_sigma_at_grid_T(name, T_K):
    """
    Прочитать один PNNL-файл молекулы при температуре T_K (должна быть в
    числе доступных) и вернуть (nu_cm_asc, sigma_cm2), nu по возрастанию.
    """
    path = _PNNL_TABLES[name][T_K]
    nu_cm, absorbance = parse_pnnl_file(path)
    sigma = pnnl_absorbance_to_sigma(absorbance)

    # Упорядочим по возрастанию ν (для надёжной интерполяции по сетке).
    order = np.argsort(nu_cm)
    return nu_cm[order], sigma[order]


def load_pnnl_sigma(name, T_target=T_EXHALE_K, allow_extrapolation=False,
                    verbose=False):
    """
    Вернуть σ(ν) молекулы при целевой температуре, интерполируя ПО T в
    каждой точке сетки.

    Алгоритм:
      1) если есть точный файл при T_target — берём его;
      2) иначе берём две ближайшие температуры, окружающие T_target
         (bracket через searchsorted), читаем их σ-спектры, приводим на
         ОБЩУЮ сетку ν (объединение узлов), и линейно интерполируем по T
         в каждой точке:
             σ(T) = σ_lo + (σ_hi − σ_lo)·(T − T_lo)/(T_hi − T_lo);
      3) если T_target вне диапазона доступных T — по умолчанию ошибка
         (это была бы экстраполяция); allow_extrapolation=True зажимает
         к крайней доступной температуре с предупреждением.

    Parameters
    ----------
    name : str
        Имя молекулы (должны быть зарегистрированы файлы; см. init_db /
        register_pnnl_files).
    T_target : float
        Целевая температура, K. По умолчанию 310 K (выдох).
    allow_extrapolation : bool
        Разрешить выход за диапазон доступных T (зажим к краю).
    verbose : bool
        Печатать диагностику интерполяции.

    Returns
    -------
    nu_cm : np.ndarray
        Сетка волновых чисел, см⁻¹, по возрастанию.
    sigma_cm2 : np.ndarray
        Сечение поглощения, см²/молекула, при T_target.
    meta : dict
        {'name', 'T_target_K', 'T_used', 'mode'} для метаданных/диагностики.
    """
    if name not in _PNNL_TABLES or not _PNNL_TABLES[name]:
        raise KeyError(
            f"PNNL: для '{name}' нет данных. Сначала init_db('pnnl_data') "
            f"или register_pnnl_files('{name}', {{T: path}})."
        )

    temps = sorted(_PNNL_TABLES[name])

    # 1) точное совпадение
    for T in temps:
        if abs(T - T_target) < 1e-6:
            nu, sig = _load_sigma_at_grid_T(name, T)
            if verbose:
                print(f"[pnnl] {name}: точный файл при {T} K")
            return nu, sig, {'name': name, 'T_target_K': T_target,
                             'T_used': (T,), 'mode': 'exact'}

    T_lo_avail = temps[0]
    T_hi_avail = temps[-1]

    # 3) вне диапазона
    if T_target < T_lo_avail or T_target > T_hi_avail:
        if not allow_extrapolation:
            raise ValueError(
                f"PNNL: T_target={T_target} K вне диапазона доступных "
                f"температур [{T_lo_avail}, {T_hi_avail}] K для '{name}'. "
                f"Это была бы экстраполяция. Добавьте файл, окружающий "
                f"{T_target} K, либо allow_extrapolation=True."
            )
        T_edge = T_lo_avail if T_target < T_lo_avail else T_hi_avail
        nu, sig = _load_sigma_at_grid_T(name, T_edge)
        if verbose:
            print(f"[pnnl] {name}: T_target={T_target} вне диапазона → "
                  f"зажато к {T_edge} K (экстраполяция запрещена по физике).")
        return nu, sig, {'name': name, 'T_target_K': T_target,
                         'T_used': (T_edge,), 'mode': 'clamped'}

    # 2) bracket: ближайшие T снизу и сверху
    idx = np.searchsorted(temps, T_target)
    T_hi = temps[idx]
    T_lo = temps[idx - 1]

    nu_lo, sig_lo = _load_sigma_at_grid_T(name, T_lo)
    nu_hi, sig_hi = _load_sigma_at_grid_T(name, T_hi)

    # Общая сетка ν: объединение узлов (так не теряем разрешение ни одного
    # из файлов). Пересечение диапазонов, чтобы не экстраполировать по ν.
    nu_min = max(nu_lo[0], nu_hi[0])
    nu_max = min(nu_lo[-1], nu_hi[-1])
    nu_common = np.union1d(nu_lo, nu_hi)
    nu_common = nu_common[(nu_common >= nu_min) & (nu_common <= nu_max)]

    sig_lo_i = np.interp(nu_common, nu_lo, sig_lo)
    sig_hi_i = np.interp(nu_common, nu_hi, sig_hi)

    # Линейная интерполяция по T в каждой точке сетки.
    w = (T_target - T_lo) / (T_hi - T_lo)
    sigma_T = sig_lo_i + (sig_hi_i - sig_lo_i) * w

    if verbose:
        print(f"[pnnl] {name}: интерполяция по T {T_lo}→{T_hi} K на "
              f"{T_target} K (w={w:.3f}), узлов сетки: {nu_common.size}")

    return nu_common, sigma_T, {'name': name, 'T_target_K': T_target,
                                'T_used': (T_lo, T_hi), 'mode': 'interp'}


def clear_cache():
    """Сбросить in-memory индекс PNNL-таблиц (файлы на диске не трогаем)."""
    _PNNL_TABLES.clear()
