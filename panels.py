"""
spectrolib.panels
=================
Биомаркерные панели — конфигурационные файлы со списком молекул,
их концентраций, литературных источников и условий измерения.

Назначение: вместо того, чтобы писать концентрации биомаркеров прямо в
коде ноутбука, держать их в файлах с понятным именем и источником.
Это даёт воспроизводимость, прозрачность для диплома и возможность
делиться панелями между экспериментами.

Поддерживаемые форматы:
    .yaml / .yml — основной (хорошо читается человеком, поддерживает
                    комментарии, иерархию, типы)
    .json        — fallback, если YAML недоступен

Пример YAML-файла:

    name: "Lung cancer biomarkers panel"
    reference: "Phillips M et al, Cancer Biomarkers 2003"
    notes: "Концентрации в выдохе пациентов с НМРЛ"

    conditions:
      T_K: 310
      p_atm: 1.0
      L_cm: 10
      diluent: {air: 1.0}

    biomarkers:
      - name: C5H10O
        c_ppb: 7.5
        wavelength_nm: 290
        source: "Phillips 2003 Table 2"
      - name: C6H12O
        c_ppb: 5.2

Все поля кроме name и c_ppb (или c_ppm) — опциональны и идут в
metadata спектра, не влияя на расчёт.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import json

from .api import GasMixture


@dataclass
class Biomarker:
    """
    Один биомаркер в панели.

    name : str
        Имя молекулы для HITRAN. Должно быть в spectrolib.hitran.MOLECULE_IDS,
        иначе add_molecule упадёт. Можно использовать произвольные имена для
        синтетических панелей (например, 'Synthetic_peak') — но тогда
        панель не будет работать с реальным HITRAN-генератором.
    c_ppb / c_ppm : float
        Концентрация. Указывается ровно одна из двух (внутри храним в ppm).
    wavelength_nm : float, optional
        Характерная длина волны линии — для справки и для автовыбора
        диапазона прибора.
    source : str, optional
        Литературный источник конкретно этой строки.
    notes : str, optional
        Любые комментарии.
    """
    name: str
    c_ppm: float
    wavelength_nm: Optional[float] = None
    source: Optional[str] = None
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Biomarker':
        """Создать Biomarker из словаря (как в YAML/JSON-файле)."""
        # Защита от YAML-ловушки: NO/YES/ON/OFF без кавычек становятся
        # bool. Имя должно быть строкой.
        name = d.get('name')
        if not isinstance(name, str):
            raise ValueError(
                f"Имя биомаркера должно быть строкой, получено {name!r} "
                f"(тип {type(name).__name__}). "
                f"Если это YAML — заключи имена типа 'NO', 'YES', 'ON', "
                f"'OFF', 'NULL' в кавычки."
            )
        # Принимаем либо c_ppm, либо c_ppb
        if 'c_ppm' in d and 'c_ppb' in d:
            raise ValueError(
                f"Биомаркер {name}: нельзя задавать "
                f"одновременно c_ppm и c_ppb"
            )
        if 'c_ppm' in d:
            c_ppm = float(d['c_ppm'])
        elif 'c_ppb' in d:
            c_ppm = float(d['c_ppb']) * 1e-3
        else:
            raise ValueError(
                f"Биомаркер {name}: нужно задать c_ppm или c_ppb"
            )
        return cls(
            name=name,
            c_ppm=c_ppm,
            wavelength_nm=d.get('wavelength_nm'),
            source=d.get('source'),
            notes=d.get('notes'),
        )

    @property
    def c_ppb(self) -> float:
        return self.c_ppm * 1e3


@dataclass
class MixturePanel:
    """
    Панель биомаркеров: набор молекул + условия измерения + метаданные.

    Поля
    ----
    name : str
        Человеко-читаемое имя панели (например, "Lung cancer panel").
    reference : str, optional
        Основной литературный источник.
    notes : str, optional
        Комментарии.
    biomarkers : list[Biomarker]
    conditions : dict
        T_K, p_atm, L_cm, diluent — условия измерения. Применяются ко
        всем биомаркерам панели (все молекулы измеряются в одной кювете).

    Методы
    ------
    to_mixture()    → GasMixture для генератора спектров
    summary()       → Markdown/строковая сводка для диплома
    scaled(K)       → новая панель с концентрациями × K (преконцентрирование)
    save(path)      → сохранить как YAML/JSON
    """
    name: str
    biomarkers: List[Biomarker]
    reference: Optional[str] = None
    notes: Optional[str] = None
    conditions: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Загрузка из файла
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path) -> 'MixturePanel':
        """
        Загрузить панель из YAML или JSON файла.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Панель не найдена: {path}")

        text = path.read_text(encoding='utf-8')
        ext = path.suffix.lower()
        if ext in ('.yaml', '.yml'):
            data = _load_yaml(text)
        elif ext == '.json':
            data = json.loads(text)
        else:
            raise ValueError(
                f"Поддерживаются .yaml/.yml/.json, получено {ext!r}"
            )

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MixturePanel':
        """Создать панель из словаря (как из YAML/JSON)."""
        if 'name' not in data:
            raise ValueError("В панели обязательно поле 'name'")
        if 'biomarkers' not in data or not data['biomarkers']:
            raise ValueError("В панели должны быть биомаркеры")

        biomarkers = [Biomarker.from_dict(b) for b in data['biomarkers']]
        return cls(
            name=data['name'],
            biomarkers=biomarkers,
            reference=data.get('reference'),
            notes=data.get('notes'),
            conditions=data.get('conditions', {}) or {},
        )

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Сохранить панель в YAML или JSON (определяется по расширению)."""
        path = Path(path)
        data = self.to_dict()
        ext = path.suffix.lower()
        if ext in ('.yaml', '.yml'):
            text = _dump_yaml(data)
        elif ext == '.json':
            text = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(
                f"Поддерживаются .yaml/.yml/.json, получено {ext!r}"
            )
        path.write_text(text, encoding='utf-8')

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация для YAML/JSON."""
        return {
            'name': self.name,
            'reference': self.reference,
            'notes': self.notes,
            'conditions': dict(self.conditions),
            'biomarkers': [
                {k: v for k, v in asdict(b).items() if v is not None}
                for b in self.biomarkers
            ],
        }

    # ------------------------------------------------------------------
    # Преобразования
    # ------------------------------------------------------------------

    def to_mixture(self, profile='voigt') -> GasMixture:
        """
        Преобразовать в GasMixture для подачи в SpectrumGenerator.

        Условия (T, p, L, diluent) берутся из self.conditions.
        Дефолты: T=310 K, p=1 атм, L=10 см, diluent={'air': 1.0}.
        """
        composition = {b.name: b.c_ppm for b in self.biomarkers}
        return GasMixture(
            composition=composition,
            T_K=float(self.conditions.get('T_K', 310.0)),
            p_atm=float(self.conditions.get('p_atm', 1.0)),
            L_cm=float(self.conditions.get('L_cm', 10.0)),
            diluent=self.conditions.get('diluent'),
            profile=profile,
        )

    def generate(self, generator, profile='voigt'):
        """
        Сгенерировать спектр одной командой.

        panel.generate(gen) эквивалентно gen.generate(panel.to_mixture()),
        но дополнительно прописывает имя панели в metadata спектра.
        Это позволяет plot() автоматически использовать имя панели как
        заголовок графика.

        Parameters
        ----------
        generator : SpectrumGenerator
        profile : {'voigt', 'lorentz', 'gauss'}

        Returns
        -------
        Spectrum
        """
        spec = generator.generate(self.to_mixture(profile=profile))
        spec.meta['panel_name'] = self.name
        spec.meta['panel_reference'] = self.reference
        spec.meta['panel_n_biomarkers'] = len(self.biomarkers)
        return spec

    def scaled(self, factor: float) -> 'MixturePanel':
        """
        Копия панели с концентрациями × factor.

        Имитация преконцентрирования (sorbent tube, SPME, cryotrap):
        фактор обогащения K_pre домножает все концентрации, остальное
        сохраняется.
        """
        new_biomarkers = [
            Biomarker(
                name=b.name,
                c_ppm=b.c_ppm * factor,
                wavelength_nm=b.wavelength_nm,
                source=b.source,
                notes=b.notes,
            )
            for b in self.biomarkers
        ]
        new_notes = (self.notes or '') + (
            f' [scaled by {factor}]' if factor != 1.0 else ''
        )
        return MixturePanel(
            name=self.name,
            biomarkers=new_biomarkers,
            reference=self.reference,
            notes=new_notes.strip(),
            conditions=dict(self.conditions),
        )

    # ------------------------------------------------------------------
    # Сводка
    # ------------------------------------------------------------------

    def summary(self, fmt: str = 'text') -> str:
        """
        Текстовая сводка панели — для лога или диплома.

        fmt : {'text', 'markdown'}
        """
        lines = []
        if fmt == 'markdown':
            lines.append(f'# {self.name}')
            if self.reference:
                lines.append(f'**Источник:** {self.reference}')
            if self.notes:
                lines.append(f'*{self.notes}*')
            lines.append('')
            if self.conditions:
                lines.append('**Условия:**')
                for k, v in self.conditions.items():
                    lines.append(f'- {k}: {v}')
                lines.append('')
            lines.append('| # | Молекула | c, ppb | c, ppm | λ, нм | Источник |')
            lines.append('|---|---|---|---|---|---|')
            for i, b in enumerate(self.biomarkers, 1):
                lam = f'{b.wavelength_nm:.1f}' if b.wavelength_nm else '—'
                src = b.source or '—'
                lines.append(
                    f'| {i} | {b.name} | {b.c_ppb:g} | {b.c_ppm:g} | '
                    f'{lam} | {src} |'
                )
        else:
            lines.append(f'Panel: {self.name}')
            if self.reference:
                lines.append(f'Reference: {self.reference}')
            if self.notes:
                lines.append(f'Notes: {self.notes}')
            if self.conditions:
                cond_str = ', '.join(f'{k}={v}' for k, v in self.conditions.items())
                lines.append(f'Conditions: {cond_str}')
            lines.append(f'Biomarkers ({len(self.biomarkers)}):')
            for b in self.biomarkers:
                lam = f'@ {b.wavelength_nm:.1f} nm' if b.wavelength_nm else ''
                src = f'  [{b.source}]' if b.source else ''
                lines.append(
                    f'  - {b.name:10s}  {b.c_ppb:>10.4g} ppb  {lam}{src}'
                )
        return '\n'.join(lines)

    def __repr__(self):
        return (f"MixturePanel({self.name!r}, "
                f"{len(self.biomarkers)} biomarkers)")

    def __len__(self):
        return len(self.biomarkers)

    def __iter__(self):
        return iter(self.biomarkers)


# ---------------------------------------------------------------------------
# Утилитарная обёртка
# ---------------------------------------------------------------------------

def load_mixture_panel(path) -> MixturePanel:
    """Шорткат: load_mixture_panel('foo.yaml') == MixturePanel.from_file(...)."""
    return MixturePanel.from_file(path)


# ---------------------------------------------------------------------------
# YAML с graceful fallback на JSON-only
# ---------------------------------------------------------------------------

def _load_yaml(text):
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "Для чтения YAML-файлов установи pyyaml: pip install pyyaml. "
            "Альтернатива — использовать .json формат."
        ) from e
    return yaml.safe_load(text)


def _dump_yaml(data):
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "Для записи YAML установи pyyaml: pip install pyyaml."
        ) from e
    return yaml.safe_dump(
        data, allow_unicode=True, sort_keys=False, default_flow_style=False
    )
