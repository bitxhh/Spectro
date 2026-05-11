"""
spectrolib.api
==============
Объектный фасад поверх fluent-класса Spectrum.

Совпадает с публичным API из спецификации диплома:
    Instrument, GasMixture, NoiseModel, SpectrumGenerator.

Пример:

    instrument = Instrument(
        wavelength_range=(750, 770),
        sampling_step=0.001,
        ils=GaussILS(fwhm=1.0),
    )
    mixture = GasMixture(
        composition={'O2': 210000, 'H2O': 10000},
        T_K=296, p_atm=1.0, L_cm=10.0,
    )
    noise = NoiseModel(thermal_sigma=0.005, shot_n_photons_max=1e4)

    gen = SpectrumGenerator(instrument, noise_model=noise, seed=42)
    spec = gen.generate(mixture)
    # spec — обычный Spectrum с заполненными метаданными.

Всё, что fluent умеет — доступно через возвращаемый Spectrum
(напр. `gen.generate(mix).add_baseline(slope=1e-4)` для пост-обработки).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

from .spectrum import Spectrum
from .ils import ILS, GaussILS
from .noise import NoiseModel


@dataclass
class Instrument:
    """
    Описание прибора.

    Параметры
    ---------
    wavelength_range : (lo, hi)
        Диапазон длин волн в нм.
    sampling_step : float
        Шаг сетки, нм.
    ils : ILS, optional
        Аппаратная функция. Если None — спектр без свёртки (идеальное
        разрешение). Полезно для отладки и для верхней границы качества.
    """
    wavelength_range: Tuple[float, float]
    sampling_step: float
    ils: Optional[ILS] = None

    def empty_spectrum(self):
        """Создать пустой Spectrum на сетке этого прибора."""
        lo, hi = self.wavelength_range
        return Spectrum.from_range(lo, hi, step_nm=self.sampling_step)


@dataclass
class GasMixture:
    """
    Описание газовой смеси.

    composition : {имя_молекулы: концентрация_в_ppm}
        Имена должны быть в spectrolib.hitran.MOLECULE_IDS.
    T_K, p_atm, L_cm — условия для всех молекул в смеси.

    diluent : dict, optional
        Состав уширяющей среды для всех молекул, передаётся в hapi.
        Дефолт {'air': 1.0} — приемлем для биомаркеров в ppm/ppb,
        для основных газов выдоха (CO₂ 4%, H₂O 5%) можно явно указать
        {'air': 0.91, 'self': 0.04, 'h2o': 0.05} — но hapi не у всех
        молекул имеет self/H₂O-broadening, проверять отдельно.

    profile : {'voigt', 'lorentz', 'gauss'}
        Профиль для всех молекул сразу. Если нужны разные — собирай
        Spectrum вручную через add_molecule.
    """
    composition: Dict[str, float]
    T_K: float = 310.0          # температура выдоха
    p_atm: float = 1.0
    L_cm: float = 10.0
    diluent: Optional[Dict[str, float]] = None
    profile: str = 'voigt'

    def with_L(self, L_cm):
        """Копия смеси с новой длиной пути. Удобно для оптимизации по L."""
        return GasMixture(
            composition=dict(self.composition),
            T_K=self.T_K, p_atm=self.p_atm, L_cm=L_cm,
            diluent=dict(self.diluent) if self.diluent else None,
            profile=self.profile,
        )

    def with_composition(self, **kwargs):
        """
        Копия смеси с обновлённой композицией. Передавай молекулы как kwargs:
            mix.with_composition(O2=210000, H2O=50000)
        Молекулы, не указанные здесь, остаются как есть. Чтобы убрать молекулу,
        передай её с None: mix.with_composition(CO2=None).
        """
        new_comp = dict(self.composition)
        for k, v in kwargs.items():
            if v is None:
                new_comp.pop(k, None)
            else:
                new_comp[k] = v
        return GasMixture(
            composition=new_comp,
            T_K=self.T_K, p_atm=self.p_atm, L_cm=self.L_cm,
            diluent=dict(self.diluent) if self.diluent else None,
            profile=self.profile,
        )


@dataclass
class SpectrumGenerator:
    """
    Сборщик Spectrum по описаниям Instrument + GasMixture + NoiseModel.

    seed управляет всей случайностью внутри (сейчас — только NoiseModel).
    Если seed=None — каждый вызов даёт новую реализацию шума.

    Внимание: воспроизводимость seed работает «по серии», а не «по
    каждому вызову». У генератора есть скрытое состояние
    `_seed_offset`, которое инкрементируется на каждый вызов
    .generate() с заданным NoiseModel и на каждую реализацию внутри
    .generate_averaged() / .snr_vs_n_realizations(). Поэтому:

    - последовательность из N вызовов `gen.generate(mix)` на одном
      и том же объекте даёт N **разных** реализаций шума, что и
      ожидается на практике (имитация нескольких независимых измерений);
    - чтобы получить тот же спектр, что и при «холодном» первом
      вызове, нужно создать новый SpectrumGenerator с тем же seed
      (offset обнулится).

    Сам набор seed-ов детерминирован и воспроизводим по seed целиком,
    просто не совпадает с тем, что было бы при «независимых» вызовах
    с одинаковым seed.
    """
    instrument: Instrument
    noise_model: Optional[NoiseModel] = None
    seed: Optional[int] = None
    # счётчик для генерации воспроизводимой последовательности при
    # многократных вызовах .generate() с одним базовым seed
    _seed_offset: int = field(default=0, init=False, repr=False)

    def generate(self, mixture: GasMixture) -> Spectrum:
        """
        Сгенерировать спектр по описанию смеси.

        Алгоритм:
            1. Пустой спектр на сетке прибора.
            2. Все молекулы смеси через add_molecule.
            3. Свёртка с ILS прибора (если задана).
            4. Применение NoiseModel (если задана).

        Возвращает обычный Spectrum, поверх которого можно делать
        дополнительные fluent-операции.
        """
        spec = self.instrument.empty_spectrum()

        for name, c_ppm in mixture.composition.items():
            spec.add_molecule(
                name, c_ppm=c_ppm, L_cm=mixture.L_cm,
                T_K=mixture.T_K, p_atm=mixture.p_atm,
                profile=mixture.profile,
                diluent=mixture.diluent,
            )

        if self.instrument.ils is not None:
            spec.convolve_ils(self.instrument.ils)

        if self.noise_model is not None:
            seed = (None if self.seed is None
                    else self.seed + self._seed_offset)
            spec.add_noise_model(self.noise_model, seed=seed)
            self._seed_offset += 1

        # Сохраняем полный контекст в meta — поверх metadata Spectrum
        spec.meta['instrument'] = {
            'wavelength_range': self.instrument.wavelength_range,
            'sampling_step': self.instrument.sampling_step,
            'ils': repr(self.instrument.ils) if self.instrument.ils else None,
        }
        spec.meta['mixture'] = {
            'composition': dict(mixture.composition),
            'T_K': mixture.T_K, 'p_atm': mixture.p_atm,
            'L_cm': mixture.L_cm,
        }
        spec.meta['generator_seed'] = self.seed
        return spec

    # -----------------------------------------------------------------
    # Усреднение нескольких реализаций («N выдохов»)
    # -----------------------------------------------------------------

    def generate_averaged(self, mixture: GasMixture,
                           n_realizations: int,
                           domain: str = 'transmittance') -> 'Spectrum':
        """
        Сгенерировать N независимых реализаций спектра и усреднить их.

        Имитация протокола измерения: «пациент делает N выдохов подряд
        с одинаковым составом, прибор регистрирует N спектров, мы их
        усредняем для повышения SNR».

        SNR для аддитивного шума в среднем растёт как √N.

        Parameters
        ----------
        mixture : GasMixture
            Описание смеси. Состав детерминирован (биологический разброс
            не моделируется в этой версии).
        n_realizations : int
            Число реализаций (N выдохов). N=1 эквивалентно generate().
        domain : {'transmittance', 'optical_depth'}
            В каком пространстве усреднять.

            'transmittance' (дефолт, физически правильно): усредняем то,
            что регистрирует детектор. Стандарт для FTIR-спектроскопии
            и любых приборов на интенсивности. При сильно нелинейном
            поглощении (T → 0) даёт небольшое смещение, но для слабых
            линий (наш случай) разница пренебрежима.

            'optical_depth' (флаг): усредняем в пространстве OD.
            Удобно, когда спектры представлены сразу в OD, и их
            структура аддитивна по молекулам. При слабом поглощении
            (OD < 0.1) практически совпадает с transmittance-режимом.

        Returns
        -------
        Spectrum
            Спектр с усреднённым наблюдаемым сигналом и теми же
            метаданными, что у одной реализации, плюс информация
            о числе усреднённых реализаций в meta['averaging'].

        Notes
        -----
        Реализация **оптимизирована**: чистая часть (HITRAN + ILS)
        считается один раз, а N раз накладывается только шум.
        Это критично, потому что HITRAN-расчёт — самый медленный шаг.
        """
        if n_realizations < 1:
            raise ValueError(f"n_realizations должно быть ≥ 1, "
                              f"получено {n_realizations}")
        if domain not in ('transmittance', 'optical_depth'):
            raise ValueError(
                f"domain должен быть 'transmittance' или 'optical_depth', "
                f"получено {domain!r}"
            )

        # Один раз считаем чистый спектр (HITRAN + ILS, без шума)
        base = self._generate_clean(mixture)

        if self.noise_model is None:
            # Без шума усреднять нечего — вернём один экземпляр
            base.meta['averaging'] = {
                'n_realizations': n_realizations,
                'domain': domain,
                'note': 'no NoiseModel — averaging trivial',
            }
            return base

        # Накапливаем сумму в выбранном пространстве
        import numpy as _np
        from .spectrum import Spectrum as _Spectrum

        accumulator = _np.zeros_like(base.wavelength_nm)

        for i in range(n_realizations):
            # Каждая реализация — копия чистого спектра + новый шум
            real = base.copy()
            seed_i = (None if self.seed is None
                       else self.seed + self._seed_offset + i)
            real.add_noise_model(self.noise_model, seed=seed_i)

            if domain == 'transmittance':
                accumulator += real.transmittance
            else:
                accumulator += real.optical_depth

        avg = accumulator / n_realizations

        # Сборка результирующего спектра
        result = base.copy()  # чистая часть та же
        if domain == 'transmittance':
            # Avg(T) → пересчёт в OD: OD_obs = -ln(T_avg);
            # шум_OD получаем как разницу с чистой OD
            T_clipped = _np.clip(avg, 1e-12, None)
            od_obs = -_np.log(T_clipped)
            # Чистая часть в base — _clean_optical_depth (без шума, без дрейфа).
            # Шумовая добавка в OD: то, что осталось сверх неё.
            # Раскладываем по полям как: _noise_T = (T_avg - T_clean),
            # _noise_OD = 0. Чтобы свойство optical_depth сошлось.
            result._noise_T = avg - base.true_transmittance
            result._noise_OD = _np.zeros_like(avg)
        else:
            # Avg(OD) → шум целиком в OD-пространстве
            result._noise_T = _np.zeros_like(avg)
            result._noise_OD = avg - base.true_optical_depth

        # Обновляем meta и noise-словарь
        result.noise = dict(self.noise_model.to_dict())
        result.noise['seed'] = self.seed
        result.noise['averaging'] = {
            'n_realizations': n_realizations, 'domain': domain,
        }
        result.history.append(
            f"generate_averaged(n={n_realizations}, domain={domain!r})"
        )
        result.meta['averaging'] = {
            'n_realizations': n_realizations, 'domain': domain,
        }

        # Сдвигаем счётчик seed, чтобы следующий вызов был «дальше»
        self._seed_offset += n_realizations

        return result

    def _generate_clean(self, mixture: GasMixture) -> 'Spectrum':
        """
        Сгенерировать только чистую (без шума) часть спектра.
        Используется внутри generate_averaged.
        """
        spec = self.instrument.empty_spectrum()
        for name, c_ppm in mixture.composition.items():
            spec.add_molecule(
                name, c_ppm=c_ppm, L_cm=mixture.L_cm,
                T_K=mixture.T_K, p_atm=mixture.p_atm,
                profile=mixture.profile,
                diluent=mixture.diluent,
            )
        if self.instrument.ils is not None:
            spec.convolve_ils(self.instrument.ils)

        # Метаданные как в обычном generate
        spec.meta['instrument'] = {
            'wavelength_range': self.instrument.wavelength_range,
            'sampling_step': self.instrument.sampling_step,
            'ils': repr(self.instrument.ils) if self.instrument.ils else None,
        }
        spec.meta['mixture'] = {
            'composition': dict(mixture.composition),
            'T_K': mixture.T_K, 'p_atm': mixture.p_atm,
            'L_cm': mixture.L_cm,
        }
        spec.meta['generator_seed'] = self.seed
        return spec

    def snr_vs_n_realizations(self, mixture: GasMixture,
                               n_values,
                               domain: str = 'transmittance',
                               n_trials: int = 5):
        """
        Зависимость SNR от числа усреднённых реализаций.

        Для каждого N в `n_values` повторяет эксперимент `n_trials` раз
        (с разными seed), вычисляет RMS остаточного шума на пустых
        участках спектра и возвращает средние значения с разбросом.

        SNR определяется как:
            SNR(N) = max|signal| / RMS(noise|N)
        где signal — чистый спектр (без шума), noise|N — разница
        между усреднённым по N реализациям и истиной.

        Parameters
        ----------
        mixture : GasMixture
        n_values : array-like of int
            Список значений N для перебора, например [1, 2, 5, 10, 20, 50, 100].
        domain : {'transmittance', 'optical_depth'}
        n_trials : int
            Число повторов эксперимента для оценки разброса
            (mean ± std SNR при одном и том же N).

        Returns
        -------
        dict с ключами:
            n_values     — np.array входных N
            snr_mean     — np.array средних SNR
            snr_std      — np.array СКО SNR между trials
            rms_mean     — np.array средних RMS остатков
            theoretical  — np.array теоретического SNR ∝ √N (норм. на N=1)
        """
        import numpy as _np

        if self.noise_model is None:
            raise ValueError(
                "snr_vs_n_realizations требует заданного NoiseModel "
                "у генератора"
            )

        n_values = _np.asarray(n_values, dtype=int)

        # Базовый чистый спектр считаем один раз
        base = self._generate_clean(mixture)

        if domain == 'transmittance':
            signal_clean = base.true_transmittance
        else:
            signal_clean = base.true_optical_depth
        signal_amplitude = float(_np.max(_np.abs(
            signal_clean - _np.mean(signal_clean)
        )))
        if signal_amplitude == 0.0:
            raise ValueError(
                "Истинный сигнал — константа (нет молекулярных линий "
                "или нет ILS-структуры). SNR не определён. "
                "Добавь молекулу в GasMixture."
            )

        snr_mean = _np.zeros(len(n_values))
        snr_std = _np.zeros(len(n_values))
        rms_mean = _np.zeros(len(n_values))

        # Базовый seed для воспроизводимости всего эксперимента
        base_seed = 0 if self.seed is None else int(self.seed)

        for i_n, N in enumerate(n_values):
            snrs = []
            rmses = []
            for trial in range(n_trials):
                # Каждая trial — независимая реализация N выдохов
                accumulator = _np.zeros_like(base.wavelength_nm)
                for k in range(N):
                    real = base.copy()
                    real.add_noise_model(
                        self.noise_model,
                        seed=base_seed + 10_000 * trial
                                + 100 * i_n + k,
                    )
                    if domain == 'transmittance':
                        accumulator += real.transmittance
                    else:
                        accumulator += real.optical_depth
                avg = accumulator / N

                noise = avg - signal_clean
                rms = float(_np.sqrt(_np.mean(noise ** 2)))
                snr = signal_amplitude / rms if rms > 0 else _np.inf
                rmses.append(rms)
                snrs.append(snr)

            snr_mean[i_n] = _np.mean(snrs)
            snr_std[i_n] = _np.std(snrs)
            rms_mean[i_n] = _np.mean(rmses)

        # Теоретический SNR ∝ √N, нормированный на первое значение
        # (поэтому даже если первая точка n_values != 1, эталон корректен).
        theoretical = snr_mean[0] * _np.sqrt(n_values / n_values[0])

        return {
            'n_values': n_values,
            'snr_mean': snr_mean,
            'snr_std': snr_std,
            'rms_mean': rms_mean,
            'theoretical': theoretical,
            'domain': domain,
        }
