# spectrolib

Синтез абсорбционных спектров выдыхаемого воздуха для задач скрининговой
диагностики. Основа — HITRAN, физически корректные шумы, разделение
«истина vs наблюдаемое», два стиля API.

**Текущая версия: 0.6.0**

## Установка

```bash
pip install hitran-api numpy scipy pandas matplotlib pytest
```

Пакет `spectrolib/` положить в путь импорта Python (или сделать
`pip install -e .` если есть `setup.py`).

## Что библиотека умеет

### Физика спектра

- Расчёт абсорбционных спектров произвольной смеси газов через HITRAN (hapi).
- Закон Бугера-Ламберта корректно реализован: на тонких смесях линейность
  по концентрации и длине пути (проверено тестами).
- Три профиля линий: Voigt (по умолчанию), Lorentz, Doppler.
- Любой диапазон в пределах HITRAN: UV-Vis, ближний ИК, средний ИК до 25 мкм.
- Поддержка ppm/ppb, мольных долей, разных T и p, произвольного diluent.

### Аппаратная функция (ILS)

Четыре варианта формы:

- **GaussILS(fwhm)** — типичная для монохроматоров со щелью.
- **LorentzILS(fwhm)** — длиннохвостовая.
- **VoigtILS(fwhm_g, fwhm_l)** — комбинированная.
- **FromFileILS(offset, intensity)** — произвольная экспериментальная форма
  из файла. Главный канал интеграции **измеренных спектров эмиссии QD/OLED**:
  получишь измерения от фотоников ВГУ → передашь массив (offset, intensity) →
  пайплайн работает с реальной формой источника без правок.

Свёртка через FFT с краевой защитой, нормировка строгая.

### Шумы

Шесть видов с физически осмысленными параметрами:

| Шум | Параметр | Куда добавляется | Зависит от сигнала? |
|---|---|---|---|
| Тепловой (Johnson) | `thermal_sigma` | T | нет |
| Дробовой (shot) | `shot_n_photons_max` | T | да, ∝ √I |
| Цветной AR(1) | `colored_sigma`, `colored_ar` | T | нет |
| 1/f-дрейф | `drift_amplitude`, `drift_n_terms` | OD | нет |
| Периодические наводки | список `(period, amp, phase)` | T | нет |
| Одиночные выбросы | `spike_rate`, `spike_amplitude_range` | T | нет |

Дробовой зависит от интенсивности — критично для радиофизической части
пайплайна (обеляющий фильтр на основе оценённой структуры шума).
Дрейф уходит в OD-пространство (медленные изменения уровня поглощения),
остальные — в transmittance (где их физически регистрирует детектор).

Воспроизводимость через seed.

### Истина vs наблюдаемое

Каждый спектр хранит и идеал (`true_optical_depth`, `true_transmittance`,
`true_absorbance`), и наблюдаемое (`optical_depth`, `transmittance`,
`absorbance`). Истина не теряется при добавлении шума — нужна для оценки
RMSE препроцессинга в эксперименте 1 диплома.

### Усреднение N выдохов (новое в 0.4.0)

```python
spec = gen.generate_averaged(mix, n_realizations=10)
```

Имитирует протокол «N выдохов с интервалом», усредняет наблюдаемый сигнал.
Физически правильно: усреднение в transmittance-пространстве (как делает
реальный FTIR), опционально — в OD через флаг `domain='optical_depth'`.

Внутри оптимизировано: HITRAN-расчёт делается **один раз**, N раз
накладывается только шум — иначе при N=1000 было бы слишком долго.

### SNR vs N (новое в 0.4.0)

```python
result = gen.snr_vs_n_realizations(mix, n_values=[1, 5, 10, 50, 100],
                                    n_trials=20)
plot_snr_vs_n(result)
```

Возвращает зависимость SNR от числа усреднённых реализаций со средним
значением и СКО по `n_trials` повторам. Закон √N подтверждается на
синтетике на 2.5 порядка по N. Это твой инструмент для оптимизации
протокола измерения: «сколько выдохов нужно для достижения порога
детектирования при заданном шуме».

### Биомаркерные панели (новое в 0.5.0)

Вместо того, чтобы держать концентрации биомаркеров в коде ноутбука,
описывай их в YAML/JSON-файлах. Прозрачность для диплома, версионирование,
переиспользование между экспериментами.

Файл `lung_cancer_panel.yaml`:

```yaml
name: "Lung cancer biomarkers panel"
reference: "Phillips M et al, Cancer Biomarkers 2003"
notes: "Концентрации в выдохе пациентов с НМРЛ"

conditions:
  T_K: 310
  p_atm: 1.0
  L_cm: 3000          # 30 м multipass
  diluent: {air: 1.0}

biomarkers:
  - name: "NO"        # внимание: NO без кавычек YAML делает False
    c_ppb: 20
    wavelength_nm: 5263
    source: "ATS/ERS 2005"
  - name: CO
    c_ppm: 1.5
    wavelength_nm: 4666
    source: "Risby 2006"
```

Использование в Python:

```python
from spectrolib import load_mixture_panel, SpectrumGenerator

panel = load_mixture_panel('lung_cancer_panel.yaml')

# Сводка для диплома (тоже доступно в markdown)
print(panel.summary(fmt='markdown'))

# Генерация спектра одной командой
spec = panel.generate(gen)
spec.plot()    # заголовок графика — имя панели

# Преконцентрирование (sorbent tube, SPME, cryotrap)
panel_pre = panel.scaled(K_pre=100)
spec_pre = panel_pre.generate(gen)
```

В каталоге `spectrolib/example_panels/` — два примера:
- `breath_demo.yaml` — основные газы выдоха (рабочий, использует только
  молекулы из MOLECULE_IDS)
- `lung_cancer_template.yaml` — шаблон для рака лёгких с указаниями,
  как его расширять (большинство альдегидов требуют добавления в
  MOLECULE_IDS)

### Многоканальная регистрация (новое в 0.6.0)

Реальный прибор на OLED/QD-пикселях не видит спектр на тонкой сетке —
он регистрирует N интегральных значений по широким полосам каналов:

```python
from spectrolib import Channel, ChannelSet, load_channel_set

# Базовый набор: 15 равномерных каналов 750-790 нм с FWHM=25 нм
channels = ChannelSet.uniform(start_nm=750, stop_nm=790, n=15, fwhm_nm=25)

# Или произвольный набор после твоей оптимизации:
channels = ChannelSet.from_centers(
    centers_nm=[745, 752, 758, 763, 770, 778, 785, 792],
    fwhm_nm=[20, 25, 22, 18, 25, 25, 30, 22],
)

# Или из YAML-файла (для воспроизводимости экспериментов):
channels = load_channel_set('qd_set_v1.yaml')

# Регистрация: тонкая сетка → N каналов
ch_spec = spec.to_channels(channels)
ch_spec.values            # np.ndarray длины N — для классификатора
ch_spec.absorbance        # то же в A
ch_spec.plot()            # столбчатая диаграмма

# Когда придут реальные QD от фотоников — измеренные спектры эмиссии
# подключаются как форма канала через FromFileILS:
from spectrolib import FromFileILS
real_qd_shape = FromFileILS(measured_offset, measured_intensity)
ch = Channel(center_nm=760, fwhm_nm=25, shape=real_qd_shape)
```

Конфиг каналов в YAML:

```yaml
name: "QD set v1: оптимизированный 15-канальный"
notes: "Результат эксперимента 5 — оптимизация под рак лёгких"
channels:
  - {center_nm: 745.0, fwhm_nm: 22.0, shape: gauss, name: ch_00}
  - {center_nm: 758.0, fwhm_nm: 25.0, shape: gauss, name: ch_01}
  # ...
```

В `spectrolib/example_channel_sets/` — пример `baseline_15ch.yaml`.

### Визуализация

Одной командой:

```python
spec.plot()                            # авто: compare если шум, иначе истина
spec.plot(kind='absorbance')           # в абсорбансе
spec.plot(kind='optical_depth')        # в OD
spec.plot_clean_vs_noisy()             # двухпанельный с RMS

plot_overlay([s1, s2, s3], labels=['L=10', 'L=50', 'L=100'])
plot_snr_vs_n(result)                  # зависимость SNR от N
```

Заголовки автоматически собираются из метаданных (молекулы, T, p, L).
Шкалы адаптируются под глубину линий. Все функции возвращают `(fig, ax)` —
стиль можно дотюнить через matplotlib.

## Два стиля API

### Fluent — для интерактивной отладки

```python
from spectrolib import Spectrum, GaussILS, NoiseModel

spec = (Spectrum.from_range(759, 767, step_nm=0.005)
        .add_molecule('O2', c_ppm=210000, L_cm=10, T_K=296, p_atm=1)
        .convolve_ils(GaussILS(fwhm=0.5))
        .add_noise_model(
            NoiseModel(thermal_sigma=0.005, shot_n_photons_max=1e4),
            seed=42))

spec.plot()
```

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
    composition={'O2': 210000, 'H2O': 50000},
    T_K=310, p_atm=1.0, L_cm=10,
)
noise = NoiseModel(thermal_sigma=0.005, shot_n_photons_max=1e4)

gen = SpectrumGenerator(inst, noise_model=noise, seed=42)
spec = gen.generate(mix)

# Оптимизация по длине кюветы
for L in [10, 50, 100, 500, 1000]:
    s = gen.generate(mix.with_L(L))
    print(f'L = {L} см: T_min = {s.true_transmittance.min():.3f}')

# Оптимизация по числу выдохов
result = gen.snr_vs_n_realizations(
    mix.with_L(100),
    n_values=[1, 5, 10, 20, 50, 100],
    n_trials=10,
)
```

Удобные шорткаты:

- `mix.with_L(new_L)` — копия с другой длиной пути
- `mix.with_composition(O2=200000)` — копия с обновлёнными концентрациями

## Структура

```
spectrolib/
├── __init__.py             — публичный API
├── physics.py              — константы (scipy.constants), единицы, БЛБ
├── hitran.py               — обёртка над hapi
├── ils.py                  — ILS (Gauss/Lorentz/Voigt/FromFile)
├── noise.py                — NoiseModel и 6 видов шума
├── spectrum.py             — главный класс Spectrum (fluent)
├── api.py                  — Instrument, GasMixture, SpectrumGenerator
├── plotting.py             — единая визуализация
├── panels.py               — MixturePanel: биомаркерные конфиги
├── channels.py             — Channel/ChannelSet/ChannelizedSpectrum
├── io.py                   — загрузка CSV/TXT/XLSX
├── example_panels/         — примеры YAML-панелей биомаркеров
├── example_channel_sets/   — примеры YAML-конфигов прибора
└── tests/                  — pytest-набор (155 тестов, ~7 с)
```

## Тесты

```bash
pytest spectrolib/tests/ -v
```

155 тестов, включая HITRAN-интеграцию через моки (без сети).

## Что планируется

**Многоканальная регистрация** (модуль 7 спецификации). Сейчас спектр —
тонкая сетка из тысяч точек, как у спектрометра высокого разрешения.
Реальный прибор с OLED/QD-каналами регистрирует не сетку, а 15-30
интегральных значений по широким полосам каналов. Архитектурно это:

- В `Instrument` добавляется конфиг каналов: список `(центр, FWHM, форма)`.
- Метод `spec.to_channels(channels)` интегрирует истинный спектр по
  полосам каналов, возвращает `ChannelizedSpectrum`.
- Шум моделируется отдельно для каждого канала.

Без этого невозможен эксперимент 2 диплома (точность классификации vs
число биомаркеров/каналов) и вытекающее ТЗ на квантовые точки.

**Сценарии заболеваний**. `DiseaseScenarioGenerator` со словарём
`{disease: {biomarker: (low, high)}}`. Реализуется тривиально, ждёт
данных от Сеченовки/Кошелева — иначе значения в словаре будут
placeholder'ами без научной ценности.

**Биологический разброс**. Сейчас «один выдох» = одна реализация шума,
смесь детерминирована. Реальные выдохи варьируются между измерениями.
Будет добавлен опциональный режим: концентрации сэмплируются из
заданных распределений на каждой реализации.

**Преконцентрирование**. Для биомаркеров рака лёгких (десятки ppb)
прямая регистрация на L=10 см невозможна. В диплом войдёт расчёт с
параметром K_pre ∈ [1, 1000], имитирующим sorbent tube или другой
preconcentrator. Технически это просто множитель на концентрации в
`GasMixture` — но имеет смысл явно вынести в API после уточнения
сценариев применения.

## Версии

- **0.6.0** — многоканальная регистрация (Channel/ChannelSet/ChannelizedSpectrum),
              спектр → каналы прибора одной командой, реальные QD как форма
              канала через FromFileILS; параметр `step_cm` в add_molecule
              против "Big wavenumber step" warning; улучшенный парсер CSV
- **0.5.0** — биомаркерные панели (YAML/JSON), панель → спектр одной командой,
              преконцентрирование через `scaled()`
- **0.4.0** — усреднение реализаций, SNR vs N
- **0.3.0** — унифицированная визуализация, объектный API оптимизации (with_L)
- **0.2.0** — разделение истина/наблюдаемое, 6 видов шума, ILS-классы,
              объектный фасад, scipy.constants
- **0.1.0** — базовый fluent API, HITRAN, гауссова ILS, простой шум
