# spectrolib

`spectrolib` — Python-библиотека для синтеза, обработки и анализа
абсорбционных спектров газовых смесей. Проект ориентирован на задачи
газоанализа выдыхаемого воздуха и проектирование компактной многоканальной
спектральной платформы, но его можно использовать и для более общих задач
моделирования показаний спектрометров.

**Текущая версия: 0.8.0**

## Что Умеет

- Синтезировать спектры газовых смесей на сетке длин волн через разные
  источники сечений поглощения: HITRAN line-by-line, PNNL IR, MPI-Mainz UV/VIS
  и HITRAN absorption cross-sections (`.xsc`).
- Сохранять единый физический хвост расчёта для всех баз данных: сечение
  `sigma [cm^2/molecule]` → закон Бугера-Ламберта-Бера → оптическая плотность
  на сетке спектра.
- Моделировать аппаратную функцию прибора: `GaussILS`, `LorentzILS`,
  `VoigtILS`, `FromFileILS`.
- Накладывать физически осмысленные шумы: тепловой, дробовой, цветной AR(1),
  1/f-дрейф, периодические наводки и выбросы.
- Хранить отдельно идеальный сигнал и наблюдаемый сигнал после ILS/шума:
  `true_transmittance`, `true_absorbance`, `true_optical_depth` и
  `transmittance`, `absorbance`, `optical_depth`.
- Описывать смеси, приборы, шумы и наборы каналов в YAML/JSON-конфигах.
- Генерировать усреднённые реализации и оценивать зависимость SNR от числа
  повторных измерений.
- Переводить тонкосеточный спектр в многоканальную регистрацию
  (`ChannelSet`, `ChannelizedSpectrum`) для OLED/QD/фильтровых приборов.
- Строить графики через единый стиль `spectrolib.plotstyle`.
- Сохранять спектры в CSV/TXT с метаданными или в NPZ-архив.
- Выполнять покомпонентный информационный аудит спектральных блоков для
  выбора каналов QD-платформы (`spectrolib.audit`).

## Установка

Из корня репозитория:

```bash
pip install -e .
```

Опциональные зависимости:

```bash
pip install -e ".[test]"   # pytest
pip install -e ".[excel]"  # openpyxl для чтения .xlsx через load_spectrum
```

Минимальная версия Python указана в `pyproject.toml`: `>=3.9`.

Файлы пакета в этом репозитории лежат в корне, а не в отдельной папке
`spectrolib/`. Это настроено через `package-dir = {"spectrolib" = "."}`.
После установки импорт обычный:

```python
from spectrolib import Spectrum, GaussILS, NoiseModel
```

## Быстрый Старт

Минимальный fluent-пример: спектр O2 в A-полосе около 760 нм, гауссова ILS и
шумовая модель.

```python
from spectrolib import Spectrum, GaussILS, NoiseModel

spec = (
    Spectrum.from_range(759, 767, step_nm=0.005)
    .add_molecule('O2', c_ppm=210_000, L_cm=10, T_K=296, p_atm=1.0)
    .convolve_ils(GaussILS(fwhm=0.5))
    .add_noise_model(
        NoiseModel(thermal_sigma=0.005, shot_n_photons_max=1e4),
        seed=42,
    )
)

fig, ax = spec.plot()
```

При первом обращении к HITRAN-молекуле HAPI скачает таблицу линий и положит её
в локальный кеш.

## Объектный API

Для пайплайнов удобнее задавать прибор, смесь и шум отдельно.

```python
from spectrolib import Instrument, GasMixture, SpectrumGenerator
from spectrolib import GaussILS, NoiseModel

inst = Instrument(
    wavelength_range=(759, 767),
    sampling_step=0.005,
    ils=GaussILS(fwhm=0.5),
)

mix = GasMixture(
    composition={'O2': 210_000, 'H2O': 50_000},
    T_K=310,
    p_atm=1.0,
    L_cm=10,
)

noise = NoiseModel(thermal_sigma=0.005, shot_n_photons_max=1e4)

gen = SpectrumGenerator(inst, noise_model=noise, seed=42)
spec = gen.generate(mix)
```

Удобные копии смеси:

- `mix.with_L(new_L)` — новая длина оптического пути.
- `mix.with_T(new_T)` — новая температура.
- `mix.with_composition(CO=1.5, NO=None)` — обновить или удалить компоненты.
- `mix.preconcentrated(K_pre)` — домножить концентрации на фактор
  преконцентрирования.

## Конфиги

### Прибор

`Instrument` можно загрузить из YAML/JSON. Пример есть в
`example_instruments/ftir_ref.yaml`.

```python
from spectrolib import load_instrument

inst = load_instrument('example_instruments/ftir_ref.yaml')
```

Схема:

```yaml
name: "FTIR reference (NIR)"
wavelength_range: [833.0, 2500.0]
sampling_step: 0.05
ils:
  type: gauss
  fwhm: 0.208
```

### Шум

`NoiseModel` тоже загружается из YAML/JSON. Пример есть в
`example_noise_models/table_2_6.yaml`.

```python
from spectrolib import load_noise_model

noise = load_noise_model('example_noise_models/table_2_6.yaml')
```

Поддерживаемые поля:

- `thermal_sigma` — аддитивный тепловой шум в transmittance.
- `shot_n_photons_max` — дробовой шум, зависящий от интенсивности.
- `colored_sigma`, `colored_ar` — цветной AR(1)-шум.
- `drift_amplitude`, `drift_n_terms` — медленный дрейф в OD.
- `periodic` — периодические наводки.
- `spike_rate`, `spike_amplitude_range` — одиночные выбросы.

### Биомаркерные Панели

Смеси можно держать в YAML/JSON-панелях. Рабочие примеры лежат в
`example_panels/`:

- `breath_demo.yaml` — демонстрационная смесь выдоха.
- `lung_cancer_control_mu0.yaml` — контрольная популяционная смесь.
- `lung_cancer_disease_mu1.yaml` — популяционная смесь для группы заболевания.
- `lung_cancer_template.yaml` — шаблон панели.

```python
from spectrolib import load_mixture_panel, preconcentrate

panel = load_mixture_panel('example_panels/lung_cancer_control_mu0.yaml')
print(panel.summary(fmt='markdown'))

panel_pre = preconcentrate(panel, K_pre=100)
spec = panel_pre.generate(gen)
```

Внутри панели концентрации задаются как `c_ppm` или `c_ppb`; в объекте они
хранятся в ppm. Поле `sources` задаёт источники сечений для молекулы.

```yaml
biomarkers:
  - name: benzene
    c_ppb: 3
    sources: [mpi, hitran_xsc]
  - name: HCHO
    c_ppb: 5
    sources: [mpi, hitran]
```

## Источники Сечений

`Spectrum.add_molecule()` и `GasMixture.sources` поддерживают четыре источника:

- `hitran` — line-by-line через HAPI, основной путь для HITRAN-молекул.
- `pnnl` — PNNL IR Database, нативные ИК-сечения VOC с интерполяцией по T.
- `mpi` — MPI-Mainz UV/VIS Spectral Atlas.
- `hitran_xsc` — HITRAN absorption cross-sections (`.xsc`), включая VOC из
  Sharpe/PNNL.

Источник выбирается в таком порядке:

1. Явный `source=` в `Spectrum.add_molecule()`.
2. Локальная карта `sources=` / `GasMixture.sources`.
3. Глобальный реестр `MOLECULE_SOURCE`.
4. Дефолт `hitran`.

Для одной молекулы можно использовать несколько источников. Это нужно, когда
УФ- и ИК-полосы одной VOC лежат в разных базах.

```python
from spectrolib import GasMixture

mix = GasMixture(
    composition={'benzene': 0.003},  # ppm
    T_K=310,
    p_atm=1.0,
    L_cm=10,
    sources={'benzene': ['mpi', 'hitran_xsc']},
)
```

Локальные данные ожидаются в папках `pnnl_data/`, `mpi_data/` и
`hitran_xsc_data/` либо регистрируются вручную через функции соответствующих
модулей: `pnnl.register_pnnl_files`, `mpi.register_mpi_file`,
`hitran_xsc.register_xsc_file`.

## Многоканальная Регистрация

Тонкий спектр можно свернуть в набор интегральных каналов прибора.

```python
from spectrolib import ChannelSet, load_channel_set

channels = ChannelSet.uniform(
    start_nm=750,
    stop_nm=790,
    n=15,
    fwhm_nm=25,
)

# Или из YAML
channels = load_channel_set('example_channel_sets/baseline_15ch.yaml')

ch_spec = spec.to_channels(channels)
ch_spec.values           # transmittance по каналам
ch_spec.absorbance       # absorbance по каналам
ch_spec.plot()
```

Для экспериментально измеренной формы канала можно передать ILS-объект,
например `FromFileILS`.

```python
from spectrolib import Channel, FromFileILS

shape = FromFileILS(measured_offset_nm, measured_intensity)
channel = Channel(center_nm=760, fwhm_nm=25, shape=shape)
```

## Усреднение И SNR

`SpectrumGenerator.generate_averaged()` считает чистую часть спектра один раз,
а затем накладывает независимые реализации шума.

```python
avg = gen.generate_averaged(mix, n_realizations=10)

result = gen.snr_vs_n_realizations(
    mix,
    n_values=[1, 5, 10, 20, 50, 100],
    n_trials=10,
)
```

По умолчанию усреднение идёт в transmittance-пространстве; можно указать
`domain='optical_depth'`.

## Визуализация

```python
spec.plot()
spec.plot(kind='absorbance')
spec.plot_clean_vs_noisy()

from spectrolib import plot_overlay, plot_snr_vs_n

plot_overlay([s1, s2, s3], labels=['L=10', 'L=50', 'L=100'])
plot_snr_vs_n(result)
```

Универсальный стиль графиков доступен отдельно:

```python
from spectrolib.plotstyle import plot, scatter, bar, hist, errorbar
from spectrolib.plotstyle import subplots, save, apply_style

ax = plot(x, y, xlabel='lambda, nm', ylabel='T', color='observed')
save(ax, 'figure.pdf')
```

## Сохранение И Загрузка

```python
spec.save('spec.csv')
spec.save('spec.csv', kind='absorbance')
spec.save('spec.npz')

from spectrolib import load_spectrum

wavelength_nm, values, meta = load_spectrum('spec.csv')
```

CSV/TXT сохраняется с `#`-шапкой и метаданными. NPZ хранит массивы
`wavelength_nm`, `observed`, `true`, `clean_optical_depth`, `noise_T`,
`noise_OD` и JSON-метаданные.

## HITRAN-Кеш

```python
from spectrolib import list_local_tables
from spectrolib.hitran import clear_cache

tables = list_local_tables()
clear_cache('O2_759-775nm')
clear_cache()
```

`clear_cache()` удаляет локальные `.data`/`.header` файлы HAPI и сбрасывает
in-memory кеш.

## Информационный Аудит

Модуль `spectrolib.audit` реализует покомпонентный информационный аудит
спектральных блоков для выбора центров каналов QD-платформы. Он строит
матрицу отклика, ковариацию шума в OD-пространстве, проектор на дополнение к
базису дрейфа и жадно выбирает каналы по `J*`-критерию.

Минимальный каркас:

```python
from spectrolib import load_mixture_panel, load_noise_model
from spectrolib.audit import LocalBlock, audit_block

panel0 = load_mixture_panel('example_panels/lung_cancer_control_mu0.yaml')
panel1 = load_mixture_panel('example_panels/lung_cancer_disease_mu1.yaml')
noise = load_noise_model('example_noise_models/table_2_6.yaml')

block = LocalBlock(
    name='HCHO_UV',
    wavelength_range_nm=(260.0, 350.0),
    targets=['HCHO'],
    interferents=['acetone', 'isoprene', 'benzene', 'toluene'],
    cv_table={
        'HCHO': 0.4,
        'acetone': 0.3,
        'isoprene': 0.3,
        'benzene': 0.5,
        'toluene': 0.5,
    },
    fwhm_nm=25.0,
    kappa=200.0,
)

result = audit_block(block, panel0, panel1, noise)
print(result.summary())
```

## Структура Репозитория

```text
.
├── pyproject.toml              # packaging, зависимости, pytest config
├── __init__.py                 # публичный API и __version__
├── spectrum.py                 # Spectrum и fluent API
├── api.py                      # Instrument, GasMixture, SpectrumGenerator
├── hitran.py                   # HAPI/HITRAN line-by-line
├── pnnl.py                     # PNNL IR loader
├── mpi.py                      # MPI-Mainz UV/VIS loader
├── hitran_xsc.py               # HITRAN .xsc loader
├── databases.py                # реестр и маршрутизация источников сечений
├── ils.py                      # ILS-модели
├── noise.py                    # NoiseModel и шумы
├── panels.py                   # MixturePanel и Biomarker
├── channels.py                 # Channel, ChannelSet, ChannelizedSpectrum
├── protocol.py                 # preconcentrate
├── audit.py                    # информационный аудит каналов
├── plotting.py                 # графики спектров
├── plotstyle.py                # общий стиль matplotlib
├── io.py                       # save/load спектров
├── example_panels/             # YAML-панели смесей
├── example_channel_sets/       # YAML-наборы каналов
├── example_instruments/        # YAML-конфиги приборов
├── example_noise_models/       # YAML-модели шума
├── scripts/                    # расчётные скрипты
└── tests/                      # pytest-набор
```

## Тесты

```bash
pytest -v
```

Сетевые обращения HITRAN в тестах мокируются там, где это требуется; для
локальных баз PNNL/MPI/HITRAN-xsc используются синтетические временные файлы
или локальные fixtures.

## История Версий

- **0.8.0** — четыре источника сечений (`hitran`, `pnnl`, `mpi`,
  `hitran_xsc`), multi-source для одной молекулы, загрузка приборов и шумовых
  моделей из YAML/JSON, `preconcentrate`, популяционные панели `mu0/mu1`,
  покомпонентный информационный аудит каналов.
- **0.7.0** — `plotstyle`, сохранение спектров в CSV/NPZ, управление кешем
  HITRAN, установка через `pyproject.toml`.
- **0.6.0** — многоканальная регистрация (`Channel`, `ChannelSet`,
  `ChannelizedSpectrum`).
- **0.5.0** — биомаркерные панели YAML/JSON и преконцентрирование через
  `MixturePanel.scaled()`.
- **0.4.0** — усреднение реализаций и `snr_vs_n_realizations`.
- **0.3.0** — унифицированная визуализация и объектный API.
- **0.2.0** — разделение истины и наблюдаемого сигнала, ILS-классы,
  расширенная модель шумов.
- **0.1.0** — базовый fluent API, HITRAN, гауссова ILS и простой шум.
