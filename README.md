# spectrolib

Python-библиотека для синтеза и обработки абсорбционных спектров газовых
смесей. Основа — база линий HITRAN, физически корректная модель шумов,
строгое разделение «истинного» и «наблюдаемого» сигнала, два стиля API.

Изначально создавалась для задач газоанализа выдыхаемого воздуха
(скрининговая диагностика по летучим биомаркерам), но не привязана к
конкретному приложению: подойдёт везде, где нужно моделировать показания
спектрометра по известному составу смеси — оптимизация параметров
прибора, валидация алгоритмов обработки, обучающие выборки для ML.

**Текущая версия: 0.7.0**

## Содержание

- [Что внутри](#что-внутри)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Возможности](#возможности)
  - [Физика спектра](#физика-спектра)
  - [Аппаратная функция (ILS)](#аппаратная-функция-ils)
  - [Модель шумов](#модель-шумов)
  - [«Истина» vs «наблюдаемое»](#истина-vs-наблюдаемое)
  - [Усреднение реализаций](#усреднение-реализаций-и-snr-vs-n)
  - [Биомаркерные панели](#биомаркерные-панели-yamljson)
  - [Многоканальная регистрация](#многоканальная-регистрация)
  - [Визуализация](#визуализация)
  - [Единый стиль графиков (`plotstyle`)](#единый-стиль-графиков-plotstyle)
  - [Сохранение и загрузка спектров](#сохранение-и-загрузка-спектров)
  - [Управление кешем HITRAN](#управление-кешем-hitran)
- [Два стиля API](#два-стиля-api)
- [Структура пакета](#структура-пакета)
- [Тесты](#тесты)
- [Дорожная карта](#дорожная-карта)
- [История версий](#история-версий)

## Что внутри

- Расчёт абсорбционных спектров произвольных смесей газов через HITRAN
  (используется официальный пакет `hitran-api` / HAPI).
- Четыре модели аппаратной функции (instrument line shape, ILS), в том
  числе загрузка произвольной экспериментально измеренной формы.
- Шесть физически осмысленных видов шума (тепловой, дробовой, цветной,
  1/f-дрейф, периодические наводки, выбросы).
- Усреднение N реализаций и расчёт зависимости SNR от N — для подбора
  протокола измерения.
- Описание смесей в декларативных YAML/JSON-файлах: вся «биология»
  отделена от кода эксперимента.
- Многоканальная регистрация: переход от тонкой спектральной сетки к
  N интегральным значениям по полосам каналов (имитация приборов на
  OLED/QD-пикселях и узкополосных фотодетекторах).
- Единая визуализация: `spec.plot()` собирает заголовки из метаданных,
  адаптирует масштабы под глубину линий, возвращает `(fig, ax)` для
  дотюнивания через matplotlib.

## Установка

Из корня репозитория:

```bash
pip install -e .
```

Зависимости (numpy, scipy, pandas, matplotlib, pyyaml, hitran-api)
подтянутся автоматически из `pyproject.toml`. Опционально:

```bash
pip install -e ".[test]"    # + pytest
pip install -e ".[excel]"   # + openpyxl (для .xlsx через load_spectrum)
```

После установки пакет можно импортировать в любом проекте:

```python
from spectrolib import Spectrum, GaussILS, NoiseModel
from spectrolib.plotstyle import plot, scatter, bar, hist, errorbar
```

При первом обращении к молекуле через HAPI её таблица будет скачана с
сервера HITRAN и закеширована локально (см. [управление кешем
HITRAN](#управление-кешем-hitran)).

## Быстрый старт

Минимальный пример — спектр поглощения O₂ в полосе A (≈760 нм) с
гауссовой ILS и небольшим шумом:

```python
from spectrolib import Spectrum, GaussILS, NoiseModel

spec = (Spectrum.from_range(759, 767, step_nm=0.005)
        .add_molecule('O2', c_ppm=210_000, L_cm=10, T_K=296, p_atm=1.0)
        .convolve_ils(GaussILS(fwhm=0.5))
        .add_noise_model(NoiseModel(thermal_sigma=0.005,
                                    shot_n_photons_max=1e4),
                         seed=42))

spec.plot()
```

`spec.transmittance`, `spec.absorbance`, `spec.optical_depth` — то, что
«увидел бы прибор». `spec.true_transmittance` и т.д. — идеальные значения
до наложения ILS+шума.

## Возможности

### Физика спектра

- Произвольная смесь газов: задаётся как словарь `{молекула: концентрация}`,
  единицы — ppm, ppb или мольная доля.
- Закон Бугера — Ламберта реализован корректно: на тонких смесях
  выполнена линейность по концентрации и длине пути (покрыто тестами).
- Три профиля линий: **Voigt** (по умолчанию), **Lorentz**, **Doppler**.
- Любой диапазон в пределах HITRAN — от UV-Vis до среднего ИК ≈ 25 мкм.
- Поддержка произвольных T, p, длины кюветы L и состава разбавителя
  (`diluent`).

### Аппаратная функция (ILS)

Реальный прибор не разрешает физическую ширину линии — он сворачивает
истинный спектр со своей аппаратной функцией. В библиотеке четыре формы:

- **`GaussILS(fwhm)`** — типичная для монохроматоров со щелью.
- **`LorentzILS(fwhm)`** — длиннохвостовая, для приборов с заметным
  вкладом крыльев.
- **`VoigtILS(fwhm_g, fwhm_l)`** — комбинация гауссовой и лоренцевой.
- **`FromFileILS(offset, intensity)`** — произвольная экспериментально
  измеренная форма. Удобный канал интеграции реальных эмиссионных
  спектров QD/OLED-источников или светофильтров: получили массивы
  `(offset, intensity)` от измерительной установки — подставили
  в пайплайн без правок кода.

Свёртка считается через FFT с защитой от краевых эффектов, ядро строго
нормируется.

### Модель шумов

Шесть видов шума с физически осмысленными параметрами:

| Шум | Параметр | Накладывается на | Зависит от сигнала? |
|---|---|---|---|
| Тепловой (Johnson) | `thermal_sigma` | T (transmittance) | нет |
| Дробовой (shot) | `shot_n_photons_max` | T | да, ∝ √I |
| Цветной AR(1) | `colored_sigma`, `colored_ar` | T | нет |
| 1/f-дрейф | `drift_amplitude`, `drift_n_terms` | OD (optical depth) | нет |
| Периодические наводки | список `(period, amp, phase)` | T | нет |
| Одиночные выбросы | `spike_rate`, `spike_amplitude_range` | T | нет |

Замечания:

- **Дробовой шум зависит от интенсивности** — это важно для алгоритмов
  обработки, опирающихся на оценку структуры шума (например, обеляющих
  фильтров).
- **1/f-дрейф уходит в пространство оптической плотности**: это
  моделирует медленный плыв уровня поглощения, как у реальных приборов
  с дрейфом базовой линии.
- Остальные шумы накладываются на коэффициент пропускания —
  там, где их физически регистрирует детектор.
- Полная воспроизводимость через `seed`.

### «Истина» vs «наблюдаемое»

Каждый объект `Spectrum` хранит **обе** версии сигнала:

- `true_transmittance`, `true_absorbance`, `true_optical_depth` —
  идеальный спектр (только физика поглощения, без ILS и шума, в случае
  fluent API — или физика+ILS в зависимости от того, на каком этапе
  пайплайна снимать; см. документацию по методам);
- `transmittance`, `absorbance`, `optical_depth` — то, что увидел бы
  реальный прибор.

Идеальный спектр не теряется при добавлении шума и ILS. Это удобно для
оценки качества алгоритмов препроцессинга (RMSE против истины),
валидации обеляющих фильтров и других задач, где нужен честный «ground
truth».

### Усреднение реализаций и SNR vs N

Многие протоколы измерения сводятся к набору N независимых регистраций
одного и того же сигнала с последующим усреднением. В библиотеке это
делается одной командой:

```python
spec = gen.generate_averaged(mix, n_realizations=10)
```

Усреднение по умолчанию происходит в transmittance-пространстве (как
делает большинство реальных FTIR-приборов), при необходимости — в OD
через флаг `domain='optical_depth'`.

Внутри оптимизировано: тяжёлый HITRAN-расчёт выполняется **один раз**,
N реализаций различаются только наложенным шумом.

Для подбора оптимального N есть готовая утилита:

```python
result = gen.snr_vs_n_realizations(mix,
                                   n_values=[1, 5, 10, 50, 100],
                                   n_trials=20)
plot_snr_vs_n(result)
```

Возвращает зависимость SNR от N с разбросом по `n_trials` повторам.
Закон √N подтверждается на синтетике на 2.5 порядка по N — удобно для
ответа на вопрос «сколько повторных измерений нужно для достижения
заданного порога детектирования».

### Биомаркерные панели (YAML/JSON)

Чтобы не держать концентрации компонентов смеси в коде ноутбука, их
можно описать в декларативном файле:

```yaml
name: "Lung cancer biomarkers panel"
reference: "Phillips M et al, Cancer Biomarkers 2003"
notes: "Концентрации в выдохе пациентов с НМРЛ"

conditions:
  T_K: 310
  p_atm: 1.0
  L_cm: 3000          # 30 м multipass-кювета
  diluent: {air: 1.0}

biomarkers:
  - name: "NO"        # внимание: NO без кавычек YAML парсит как False
    c_ppb: 20
    wavelength_nm: 5263
    source: "ATS/ERS 2005"
  - name: CO
    c_ppm: 1.5
    wavelength_nm: 4666
    source: "Risby 2006"
```

Использование:

```python
from spectrolib import load_mixture_panel, SpectrumGenerator

panel = load_mixture_panel('lung_cancer_panel.yaml')

# Сводка панели в человекочитаемом виде
print(panel.summary(fmt='markdown'))

# Генерация спектра одной командой
spec = panel.generate(gen)
spec.plot()        # заголовок графика берётся из имени панели

# Имитация преконцентрирования (sorbent tube, SPME, cryotrap):
# умножаем все концентрации на K_pre
panel_pre = panel.scaled(K_pre=100)
spec_pre = panel_pre.generate(gen)
```

В каталоге `spectrolib/example_panels/` лежат два примера:

- `breath_demo.yaml` — основные газы выдоха (рабочий, использует только
  молекулы из встроенного `MOLECULE_IDS`);
- `lung_cancer_template.yaml` — шаблон с биомаркерами рака лёгких и
  указаниями, какие молекулы предварительно нужно добавить в
  `MOLECULE_IDS`.

### Многоканальная регистрация

Реальный прибор на узкополосных пиксельных фотоприёмниках (OLED, QD,
светофильтр + фотодиод) не видит спектр на тонкой сетке — он
регистрирует N интегральных значений по широким полосам каналов.
Библиотека умеет переходить от тонкой сетки к каналам:

```python
from spectrolib import Channel, ChannelSet, load_channel_set

# Равномерный набор каналов
channels = ChannelSet.uniform(start_nm=750, stop_nm=790,
                              n=15, fwhm_nm=25)

# Произвольный набор (например, после оптимизации центров)
channels = ChannelSet.from_centers(
    centers_nm=[745, 752, 758, 763, 770, 778, 785, 792],
    fwhm_nm=[20, 25, 22, 18, 25, 25, 30, 22],
)

# Из YAML-файла — для воспроизводимости экспериментов
channels = load_channel_set('qd_set_v1.yaml')

# Регистрация: тонкая сетка → N каналов
ch_spec = spec.to_channels(channels)
ch_spec.values            # np.ndarray длины N — фичи для классификатора
ch_spec.absorbance        # то же в шкале поглощения
ch_spec.plot()            # столбчатая диаграмма
```

Когда от партнёров приходят измеренные спектры эмиссии реальных
источников (QD, OLED) или пропускания фильтров, они подключаются как
форма канала через `FromFileILS`:

```python
from spectrolib import FromFileILS

real_qd_shape = FromFileILS(measured_offset, measured_intensity)
ch = Channel(center_nm=760, fwhm_nm=25, shape=real_qd_shape)
```

Пример конфига каналов:

```yaml
name: "QD set v1: оптимизированный 15-канальный"
notes: "Результат оптимизации под целевую задачу"
channels:
  - {center_nm: 745.0, fwhm_nm: 22.0, shape: gauss, name: ch_00}
  - {center_nm: 758.0, fwhm_nm: 25.0, shape: gauss, name: ch_01}
  # ...
```

В `spectrolib/example_channel_sets/` лежит готовый пример
`baseline_15ch.yaml`.

### Визуализация

Одной командой:

```python
spec.plot()                            # авто: compare если есть шум, иначе истина
spec.plot(kind='absorbance')           # в абсорбансе
spec.plot(kind='optical_depth')        # в оптической плотности
spec.plot_clean_vs_noisy()             # двухпанельный график с RMS

plot_overlay([s1, s2, s3], labels=['L=10', 'L=50', 'L=100'])
plot_snr_vs_n(result)                  # зависимость SNR от N
```

Заголовки графиков автоматически собираются из метаданных (молекулы,
температура, давление, длина кюветы). Масштабы по Y адаптируются под
глубину линий. Все функции возвращают пару `(fig, ax)`, так что стиль
можно дотюнить штатными средствами matplotlib.

### Единый стиль графиков (`plotstyle`)

`spectrolib.plotstyle` — самодостаточный модуль со стилизованными
обёртками над matplotlib. Импортируется отдельно и пригоден для любых
других проектов, не только спектров:

```python
from spectrolib.plotstyle import plot, scatter, bar, hist, errorbar
from spectrolib.plotstyle import subplots, save, apply_style
from spectrolib.plotstyle import PALETTE, SEMANTIC, MARKERS, DEFAULTS

plot(x, y, xlabel='λ, нм', ylabel='T', title='Spectrum',
     label='наблюдение', color='accent', log_y=True)
```

Все функции — `plot/scatter/bar/hist/errorbar` — принимают одинаковый
набор удобных параметров: `xlabel/ylabel/title`, `label` (легенда
включается автоматически), `color/marker/linestyle`, `log_x/log_y`,
`figsize`, `ax` (для составных графиков). Возвращают `ax`
(`ax.figure` даёт фигуру). Стиль (сетка, minor ticks, шрифты,
обрезанные верх/правая рамки) применяется автоматически.

**Палитра** — Okabe-Ito, безопасная для дальтоников. Цвет можно задать
именем (`'primary'`, `'accent'`, `'success'`, ...) или семантически
(`'observed'`, `'true'`, `'diff'`, `'theory'`):

```python
plot(x, y_obs, color='observed')
plot(x, y_true, color='true', linestyle='--', ax=ax)
```

**Сетка подграфиков** одной командой со стилем, применённым ко всем
осям, и адаптивным размером:

```python
fig, axes = subplots(2, 3)              # 6 осей, удобный итератор
for ax, data in zip(axes, datasets):
    plot(data.x, data.y, ax=ax)
```

**Сохранение в LaTeX-качестве** — вектор по дефолту:

```python
ax = plot(x, y, xlabel='λ, нм', ylabel='T')
save(ax, 'fig.pdf')                     # PDF — для LaTeX (без потери качества)
save(ax, 'fig.png')                     # PNG — 300 dpi из DEFAULTS
```

**Дефолты** (`DEFAULTS`) — мутабельный dict; меняйте под себя одной
строкой в начале ноутбука:

```python
from spectrolib import plotstyle
plotstyle.DEFAULTS['figsize'] = (10, 5)
plotstyle.DEFAULTS['savefig_dpi'] = 600
```

**LaTeX-режим** через `apply_style(preset='latex')` — подменяет
шрифты на Computer Modern и включает `text.usetex` (нужен
установленный LaTeX). Для тезисов без жёсткой типографики есть
preset `'mathtext'` — Computer Modern через mathtext без зависимости
от внешнего LaTeX:

```python
from spectrolib.plotstyle import apply_style

# В начале ноутбука: применяет дефолты ко всем последующим plt.plot
apply_style(preset='mathtext')
# Откат при желании: prev = apply_style(...); plt.rcParams.update(prev)
```

`spec.plot()` и остальные функции из `spectrolib.plotting` уже
используют этот же стиль внутри — графики из библиотеки и ваши
собственные через `plotstyle` выглядят одинаково.

### Сохранение и загрузка спектров

```python
spec.save('spec.csv')                    # CSV с метаданными в шапке
spec.save('spec.csv', kind='absorbance') # любая величина
spec.save('archive.npz')                 # полный архив (все массивы + JSON-meta)
spec.save('spec.csv', include_true=False) # без колонки эталона (для load_spectrum)
```

CSV-шапка содержит `# kind:`, `# molecules:`, `# T_K/p_atm/L_cm:`,
условия ILS/шумов и полный JSON-блок `metadata_json:` для машинного
восстановления контекста. Совместим с `load_spectrum()`:

```python
from spectrolib import load_spectrum
wl, vals, meta = load_spectrum('spec.csv')
print(meta['kind'])      # 'transmittance'
```

NPZ — для долгого хранения: внутри `wavelength_nm`, `observed`,
`true`, `clean_optical_depth`, `noise_T`, `noise_OD` и полная
`metadata_json` (JSON-строка с молекулами, ILS, шумами, историей).

### Управление кешем HITRAN

При первом обращении к молекуле HAPI скачивает таблицу линий и
сохраняет её локально. Эти таблицы можно листать и чистить:

```python
from spectrolib import list_local_tables
from spectrolib.hitran import clear_cache

list_local_tables()                    # имена всех закешированных таблиц
clear_cache('O2_759-775nm')            # удалить одну таблицу
clear_cache()                          # снести весь кеш (для перекачки с нуля)
```

`clear_cache()` удаляет файлы `.data`/`.header` с диска **и** сбрасывает
in-memory кеш HAPI — следующий `fetch_molecule` гарантированно сходит
на сервер заново.

## Два стиля API

Оба стиля живут в одной и той же кодовой базе, можно смешивать.

### Fluent — для интерактивной отладки в ноутбуке

```python
from spectrolib import Spectrum, GaussILS, NoiseModel

spec = (Spectrum.from_range(759, 767, step_nm=0.005)
        .add_molecule('O2', c_ppm=210_000, L_cm=10, T_K=296, p_atm=1)
        .convolve_ils(GaussILS(fwhm=0.5))
        .add_noise_model(
            NoiseModel(thermal_sigma=0.005, shot_n_photons_max=1e4),
            seed=42))

spec.plot()
```

Цепочечный стиль удобен, когда вы собираете и крутите один спектр
руками — добавили молекулу, посмотрели, поменяли ILS, посмотрели,
наложили шум, посмотрели.

### Объектный — для пайплайнов и оптимизации

```python
from spectrolib import (
    Instrument, GasMixture, NoiseModel, SpectrumGenerator, GaussILS,
)

inst = Instrument(
    wavelength_range=(759, 767),
    sampling_step=0.005,
    ils=GaussILS(fwhm=0.5),
)
mix = GasMixture(
    composition={'O2': 210_000, 'H2O': 50_000},
    T_K=310, p_atm=1.0, L_cm=10,
)
noise = NoiseModel(thermal_sigma=0.005, shot_n_photons_max=1e4)

gen = SpectrumGenerator(inst, noise_model=noise, seed=42)
spec = gen.generate(mix)

# Параметрическая развёртка по длине кюветы
for L in [10, 50, 100, 500, 1000]:
    s = gen.generate(mix.with_L(L))
    print(f'L = {L:>4} см: T_min = {s.true_transmittance.min():.3f}')

# Подбор числа усреднений
result = gen.snr_vs_n_realizations(
    mix.with_L(100),
    n_values=[1, 5, 10, 20, 50, 100],
    n_trials=10,
)
```

Объектный стиль рассчитан на повторяющиеся вычисления: один и тот же
прибор (`Instrument`) применяется к разным смесям, разные шумы
накладываются на один и тот же истинный спектр и т.д.

Удобные шорткаты для иммутабельных копий:

- `mix.with_L(new_L)` — новая смесь с другой длиной пути;
- `mix.with_composition(O2=200_000)` — копия с обновлённым составом.

## Структура пакета

```
spectrolib/
├── pyproject.toml          — packaging (pip install -e .)
├── __init__.py             — публичный API
├── physics.py              — константы (scipy.constants), единицы, Бугер-Ламберт
├── hitran.py               — обёртка над HAPI + clear_cache
├── ils.py                  — ILS (Gauss/Lorentz/Voigt/FromFile)
├── noise.py                — NoiseModel и 6 видов шума
├── spectrum.py             — главный класс Spectrum (fluent)
├── api.py                  — Instrument, GasMixture, SpectrumGenerator
├── plotting.py             — спектр-специфичные графики
├── plotstyle.py            — универсальный модуль стиля + plot/scatter/bar/hist
├── panels.py               — MixturePanel: декларативные конфиги смесей
├── channels.py             — Channel/ChannelSet/ChannelizedSpectrum
├── io.py                   — load_spectrum / save_spectrum (CSV/TXT/XLSX/NPZ)
├── example_panels/         — примеры YAML-панелей
├── example_channel_sets/   — примеры YAML-конфигов каналов
└── tests/                  — pytest-набор
```

## Тесты

```bash
pytest spectrolib/tests/ -v
```

Около 155 тестов, проход ~7 секунд. HITRAN-интеграция покрыта моками,
сеть для прогона тестов не нужна.

## Дорожная карта

**Сценарии и биологический разброс.** Сейчас «одно измерение» — это
одна реализация шума при детерминированной смеси. На практике состав
смеси сам по себе варьируется от измерения к измерению (например, между
выдохами одного человека). Планируется опциональный режим, где
концентрации сэмплируются из заданных распределений на каждой
реализации, плюс хелпер `DiseaseScenarioGenerator` для готовых сценариев
вида «смесь биомаркеров с диапазонами `(low, high)` по каждому
компоненту».

**Преконцентрирование как отдельный параметр API.** Технически это уже
поддерживается через `panel.scaled(K_pre=...)` и ручное умножение
концентраций. Имеет смысл вынести в явное поле `Instrument` или
`Protocol`, когда устоится перечень типичных схем (sorbent tube, SPME,
cryotrap) и их характерных коэффициентов.

**Более широкая молекулярная база.** Список `MOLECULE_IDS` сейчас
покрывает основные газы выдоха и некоторые биомаркеры. По мере
расширения сценариев применения он будет дополняться (в первую очередь —
короткоцепочечные альдегиды и кетоны).

## История версий

- **0.7.0** — модуль `plotstyle` (palette Okabe-Ito, plot/scatter/bar/
  hist/errorbar с едиными дефолтами, `subplots()`, `save()`,
  preset-ы `apply_style('mathtext'/'latex')`); `Spectrum.save()` и
  `save_spectrum()` (CSV с метаданными в шапке / NPZ-архив); `clear_cache()`
  для управления локальными таблицами HITRAN; `pyproject.toml` —
  установка через `pip install -e .`.
- **0.6.0** — многоканальная регистрация
  (`Channel` / `ChannelSet` / `ChannelizedSpectrum`); переход
  «тонкая сетка → каналы прибора» одной командой; реальные QD как
  форма канала через `FromFileILS`; параметр `step_cm` в `add_molecule`
  против предупреждения «Big wavenumber step»; улучшенный парсер CSV.
- **0.5.0** — биомаркерные панели (YAML/JSON), генерация спектра из
  панели одной командой, преконцентрирование через `scaled()`.
- **0.4.0** — усреднение реализаций (`generate_averaged`),
  `snr_vs_n_realizations`.
- **0.3.0** — унифицированная визуализация, объектный API с
  иммутабельными копиями (`with_L`, `with_composition`).
- **0.2.0** — разделение «истина / наблюдаемое», 6 видов шума,
  ILS-классы, объектный фасад, `scipy.constants`.
- **0.1.0** — базовый fluent API, HITRAN, гауссова ILS, простой шум.
