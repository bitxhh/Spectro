"""
spectrolib.protocol
===================
Этапы пробоподготовки и протокол измерения, выраженные функционально.

Сейчас здесь живёт только преконцентрирование. Логически это часть
протокола измерения (sorbent tube / SPME / криотрап перед детектором),
а не свойство смеси или панели — поэтому функция вынесена сюда.

Можно применять как к GasMixture (объектный фасад), так и к
MixturePanel (декларативные конфиги) — диспетчеризация по типу.

Пример:

    from spectrolib import GasMixture, preconcentrate

    breath = GasMixture(composition={'NO': 0.025}, T_K=310, L_cm=10)
    sample = preconcentrate(breath, K_pre=1000)   # 1 ppb → 1 ppm
    spec   = gen.generate(sample)
"""

from __future__ import annotations

from .api import GasMixture


def preconcentrate(target, K_pre):
    """
    Применить преконцентрирование с фактором обогащения K_pre.

    Параметры
    ---------
    target : GasMixture | MixturePanel
        Объект, концентрации которого нужно домножить на K_pre.
    K_pre : float
        Безразмерный фактор обогащения, K_pre > 0.
        Типичные значения для сорбентных трубок и SPME: 10²…10⁴.

    Returns
    -------
    Того же типа, что и target. Возвращает новый объект, исходный не меняется.

    Замечания
    ---------
    K_pre влияет только на концентрации. Температура, давление, длина пути
    и diluent не меняются — для них есть .with_T() / .with_L() (на GasMixture)
    или conditions (на MixturePanel).

    Если ты моделируешь параллельно и термодесорбцию с потерями, и
    обогащение по концентрации — учитывай эти потери в K_pre заранее
    (эффективный K_pre = K_geom · η_recovery).
    """
    if isinstance(target, GasMixture):
        return target.preconcentrated(K_pre)

    # MixturePanel импортируем лениво, чтобы избежать циклической зависимости
    from .panels import MixturePanel
    if isinstance(target, MixturePanel):
        return target.scaled(K_pre)

    raise TypeError(
        f"preconcentrate() ожидает GasMixture или MixturePanel, "
        f"получено {type(target).__name__}"
    )
