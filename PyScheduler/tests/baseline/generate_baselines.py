#!/usr/bin/env python3
"""Генерация/обновление baseline-снимков M0 (SPEC-FEAT-SCH-008, раздел 9).

Команда (из корня репозитория):
    python PyScheduler/tests/baseline/generate_baselines.py            # все
    python PyScheduler/tests/baseline/generate_baselines.py golden_tti5 wb_RoundRobin

Снимки обновляются после каждого слияния PR (gate 8.1): регенерировать на
новой dev-ревизии и зафиксировать обновление отдельным коммитом с указанием
ревизии в meta.dev_revision.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent.parent          # .../PyScheduler
for p in (str(PKG_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline_scenarios import BASELINE_DIR, SCENARIOS  # noqa: E402


def write_snapshot(name, data):
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"{name}.json"
    path.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False),
                    encoding="utf-8")
    return path


def main(argv):
    names = argv or sorted(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        print(f"Неизвестные сценарии: {unknown}\nДоступны: {sorted(SCENARIOS)}")
        return 1

    for name in names:
        data = SCENARIOS[name]()
        path = write_snapshot(name, data)
        n_tti = len(data["ttis"])
        print(f"[OK] {name}: {n_tti} TTI -> {path.name} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
