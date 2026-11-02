"""Правила сравнения baseline-снимков (SPEC-FEAT-SCH-008, раздел 8.2).

- Логические/дискретные значения (списки PRB/RBG, UE, QCI/LCID grants, CCE,
  offsets, переданные байты/биты) сравниваются ТОЧНО.
- float сравнивается с относительным допуском FLOAT_REL_TOL: снимок и проверка
  выполняются одним интерпретатором на одной машине, арифметика
  детерминирована, поэтому допуск нужен только против представления JSON.
  Широкий epsilon запрещён (R2) — изменение алгоритма допуском не скрывается.
- wall-clock поля (sch_total_time_us и пр.) в снимки не включаются.
"""

FLOAT_REL_TOL = 1e-9


def compare(expected, actual, path="$"):
    """Рекурсивное сравнение; возвращает список человекочитаемых отличий."""
    diffs = []

    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            diffs.append(f"{path}: expected {expected!r}, got {actual!r}")
        return diffs

    if isinstance(expected, float) or isinstance(actual, float):
        try:
            e, a = float(expected), float(actual)
        except (TypeError, ValueError):
            diffs.append(f"{path}: type mismatch {type(expected).__name__} "
                         f"vs {type(actual).__name__} ({expected!r} vs {actual!r})")
            return diffs
        if e == a:
            return diffs
        denom = max(abs(e), abs(a))
        if denom > 0 and abs(e - a) / denom <= FLOAT_REL_TOL:
            return diffs
        diffs.append(f"{path}: float expected {e!r}, got {a!r} "
                     f"(rel diff {abs(e - a) / denom if denom else 0:.3e} > {FLOAT_REL_TOL})")
        return diffs

    if isinstance(expected, (int, str)) or expected is None:
        if type(expected) is not type(actual) or expected != actual:
            diffs.append(f"{path}: expected {expected!r}, got {actual!r}")
        return diffs

    if isinstance(expected, list):
        if not isinstance(actual, list):
            diffs.append(f"{path}: expected list, got {type(actual).__name__}")
            return diffs
        if len(expected) != len(actual):
            diffs.append(f"{path}: length expected {len(expected)}, got {len(actual)}")
        for i, (e, a) in enumerate(zip(expected, actual, strict=False)):
            diffs.extend(compare(e, a, f"{path}[{i}]"))
        return diffs

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            diffs.append(f"{path}: expected dict, got {type(actual).__name__}")
            return diffs
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing:
            diffs.append(f"{path}: missing keys {missing}")
        if extra:
            diffs.append(f"{path}: unexpected keys {extra}")
        for k in sorted(set(expected) & set(actual)):
            diffs.extend(compare(expected[k], actual[k], f"{path}.{k}"))
        return diffs

    if expected != actual:
        diffs.append(f"{path}: expected {expected!r}, got {actual!r}")
    return diffs
