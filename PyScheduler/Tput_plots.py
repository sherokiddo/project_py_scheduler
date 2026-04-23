import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np


# =========================
# Настройки
# =========================
detailed_csv_file = "emp_stats_detailed.csv"
cell_csv_file = "emp_stats.csv"

window_ms = 100         # размер окна в миллисекундах
tti_duration_ms = 1     # длительность одного TTI, обычно 1 мс в LTE


# =========================
# Загрузка данных
# =========================
df = pd.read_csv(detailed_csv_file, sep=';')
df.columns = df.columns.str.strip()

cell_df = pd.read_csv(cell_csv_file, sep=';')
cell_df.columns = cell_df.columns.str.strip()


# =========================
# Подготовка данных
# =========================
df["time_ms"] = df["tti"] * tti_duration_ms
df["window_id"] = (df["time_ms"] // window_ms).astype(int)

cell_df["time_ms"] = cell_df["tti"] * tti_duration_ms
cell_df["window_id"] = (cell_df["time_ms"] // window_ms).astype(int)

window_sec = window_ms / 1000.0
simulation_time_sec = df["tti"].nunique() * tti_duration_ms / 1000.0


# =========================
# Общая скорость по окнам
# =========================
total_window = (
    df.groupby(["ue_id", "window_id"], as_index=False)["bits_transmitted_per_qci"]
    .sum()
)
total_window["throughput_mbps"] = (
    total_window["bits_transmitted_per_qci"] / window_sec / 1e6
)
total_window["time_ms"] = total_window["window_id"] * window_ms

qci_window = (
    df.groupby(["ue_id", "window_id", "qci"], as_index=False)["bits_transmitted_per_qci"]
    .sum()
)
qci_window["throughput_mbps"] = (
    qci_window["bits_transmitted_per_qci"] / window_sec / 1e6
)
qci_window["time_ms"] = qci_window["window_id"] * window_ms


# =========================
# Средняя пропускная способность за всё время
# =========================
total_avg = (
    df.groupby("ue_id", as_index=False)["bits_transmitted_per_qci"]
    .sum()
)
total_avg["avg_throughput_mbps"] = (
    total_avg["bits_transmitted_per_qci"] / simulation_time_sec / 1e6
)

qci_avg = (
    df.groupby(["ue_id", "qci"], as_index=False)["bits_transmitted_per_qci"]
    .sum()
)
qci_avg["avg_throughput_mbps"] = (
    qci_avg["bits_transmitted_per_qci"] / simulation_time_sec / 1e6
)


# =========================
# PDB success ratio
# =========================
pdb_total = (
    df.groupby("ue_id", as_index=False)[
        ["packets_extracted_late_per_qci", "packets_extracted_per_qci"]
    ].sum()
)

pdb_total["pdb_success_ratio"] = (
    (
        pdb_total["packets_extracted_per_qci"] -
        pdb_total["packets_extracted_late_per_qci"]
    ) / pdb_total["packets_extracted_per_qci"].replace(0, pd.NA)
).fillna(0)

pdb_qci = (
    df.groupby(["ue_id", "qci"], as_index=False)[
        ["packets_extracted_late_per_qci", "packets_extracted_per_qci"]
    ].sum()
)

pdb_qci["pdb_success_ratio"] = (
    (
        pdb_qci["packets_extracted_per_qci"] -
        pdb_qci["packets_extracted_late_per_qci"]
    ) / pdb_qci["packets_extracted_per_qci"].replace(0, pd.NA)
).fillna(0)

# =========================
# Expired packet ratio
# =========================
expired_total = (
    df.groupby("ue_id", as_index=False)[
        ["packets_added_per_qci", "packets_expired_per_qci"]
    ].sum()
)

expired_total["expired_ratio"] = (
    expired_total["packets_expired_per_qci"]
    / expired_total["packets_added_per_qci"].replace(0, pd.NA)
).fillna(0)

expired_qci = (
    df.groupby(["ue_id", "qci"], as_index=False)[
        ["packets_added_per_qci", "packets_expired_per_qci"]
    ].sum()
)

expired_qci["expired_ratio"] = (
    expired_qci["packets_expired_per_qci"]
    / expired_qci["packets_added_per_qci"].replace(0, pd.NA)
).fillna(0)


# =========================
# Throughput соты
# =========================
cell_window = (
    cell_df.groupby("window_id", as_index=False)["dl_transmitted_bits_sum_tti"]
    .sum()
)
cell_window["throughput_mbps"] = (
    cell_window["dl_transmitted_bits_sum_tti"] / window_sec / 1e6
)
cell_window["time_ms"] = cell_window["window_id"] * window_ms

cell_total_bits = cell_df["dl_transmitted_bits_sum_tti"].sum()
cell_avg_throughput_mbps = cell_total_bits / simulation_time_sec / 1e6


# =========================
# Параметры subplot
# =========================
ue_ids = sorted(df["ue_id"].unique())
n_ue = len(ue_ids)

ncols = min(2, n_ue)
nrows = math.ceil(n_ue / ncols)


# =========================================================
# FIGURE 1: Throughput vs Time
# =========================================================
fig1, axes1 = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes1 = np.atleast_1d(axes1).flatten()

for i, ue_id in enumerate(ue_ids):
    ax = axes1[i]

    total_ue = total_window[total_window["ue_id"] == ue_id]
    ax.plot(
        total_ue["time_ms"],
        total_ue["throughput_mbps"],
        label="Total throughput",
        linewidth=2
    )

    qci_ue = qci_window[qci_window["ue_id"] == ue_id]
    qci_list = sorted(qci_ue["qci"].unique())

    for qci in qci_list:
        qci_data = qci_ue[qci_ue["qci"] == qci]
        ax.plot(
            qci_data["time_ms"],
            qci_data["throughput_mbps"],
            label=f"QCI {qci}",
            linestyle="--"
        )

    ax.set_title(f"UE {ue_id}: Throughput vs Time")
    ax.set_xlabel("Time, ms")
    ax.set_ylabel("Throughput, Mbit/s")
    ax.grid(True)
    ax.legend()

for j in range(len(ue_ids), len(axes1)):
    fig1.delaxes(axes1[j])

fig1.tight_layout()
plt.show()


# =========================================================
# FIGURE 2: Average Throughput
# =========================================================
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes2 = np.atleast_1d(axes2).flatten()

for i, ue_id in enumerate(ue_ids):
    ax = axes2[i]

    labels = ["Total"]
    values = [
        total_avg.loc[total_avg["ue_id"] == ue_id, "avg_throughput_mbps"].iloc[0]
    ]

    qci_avg_ue = qci_avg[qci_avg["ue_id"] == ue_id].sort_values("qci")

    for _, row in qci_avg_ue.iterrows():
        labels.append(f"QCI {int(row['qci'])}")
        values.append(row["avg_throughput_mbps"])

    bars = ax.bar(labels, values, zorder=3)

    ax.set_ylabel("Пропускная способность, Мбит/с")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", zorder=0)

    ymax = max(values) if values else 0
    text_offset = ymax * 0.03 if ymax > 0 else 0.001
    ax.set_ylim(0, ymax + text_offset * 5)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + text_offset,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

for j in range(len(ue_ids), len(axes2)):
    fig2.delaxes(axes2[j])

fig2.tight_layout()
plt.show()


# =========================================================
# FIGURE 3: PDB Success Ratio
# =========================================================
fig3, axes3 = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes3 = np.atleast_1d(axes3).flatten()

for i, ue_id in enumerate(ue_ids):
    ax = axes3[i]

    pdb_labels = ["Total"]
    pdb_values = [
        pdb_total.loc[pdb_total["ue_id"] == ue_id, "pdb_success_ratio"].iloc[0]
    ]

    pdb_qci_ue = pdb_qci[pdb_qci["ue_id"] == ue_id].sort_values("qci")

    for _, row in pdb_qci_ue.iterrows():
        pdb_labels.append(f"QCI {int(row['qci'])}")
        pdb_values.append(row["pdb_success_ratio"])

    bars = ax.bar(pdb_labels, pdb_values, zorder=3)

    ax.set_title(f"UE {ue_id}: PDB Success Ratio")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Successful packets / Total packets")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", zorder=0)

    ymax = max(pdb_values) if pdb_values else 0
    text_offset = ymax * 0.03 if ymax > 0 else 0.01
    ax.set_ylim(0, 1 + text_offset * 5)

    for bar, value in zip(bars, pdb_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + text_offset,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

for j in range(len(ue_ids), len(axes3)):
    fig3.delaxes(axes3[j])

fig3.tight_layout()
plt.show()


# =========================================================
# FIGURE 4: Cell Throughput
# =========================================================
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

# Слева: throughput соты во времени
axes4[0].plot(
    cell_window["time_ms"],
    cell_window["throughput_mbps"],
    linewidth=2,
    label="Cell throughput"
)
axes4[0].set_title("Cell Throughput vs Time")
axes4[0].set_xlabel("Time, ms")
axes4[0].set_ylabel("Throughput, Mbit/s")
axes4[0].grid(True)
axes4[0].legend()

# Справа: средняя throughput соты
bars = axes4[1].bar(["Cell"], [cell_avg_throughput_mbps], zorder=3)

axes4[1].set_title("Average Cell Throughput")
axes4[1].set_xlabel("Metric")
axes4[1].set_ylabel("Average Throughput, Mbit/s")
axes4[1].set_axisbelow(True)
axes4[1].grid(True, axis="y", zorder=0)

ymax = cell_avg_throughput_mbps
text_offset = ymax * 0.03 if ymax > 0 else 0.001
axes4[1].set_ylim(0, ymax + text_offset * 5)

for bar in bars:
    value = bar.get_height()
    axes4[1].text(
        bar.get_x() + bar.get_width() / 2,
        value + text_offset,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

fig4.tight_layout()
plt.show()

# =========================================================
# FIGURE 5: Expired Packet Ratio
# =========================================================
fig5, axes5 = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes5 = np.atleast_1d(axes5).flatten()

for i, ue_id in enumerate(ue_ids):
    ax = axes5[i]

    expired_labels = ["Total"]
    expired_values = [
        expired_total.loc[
            expired_total["ue_id"] == ue_id, "expired_ratio"
        ].iloc[0]
    ]

    expired_qci_ue = expired_qci[expired_qci["ue_id"] == ue_id].sort_values("qci")

    for _, row in expired_qci_ue.iterrows():
        expired_labels.append(f"QCI {int(row['qci'])}")
        expired_values.append(row["expired_ratio"])

    bars = ax.bar(expired_labels, expired_values, zorder=3)

    ax.set_title(f"UE {ue_id}: Expired Packet Ratio")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Expired packets / Added packets")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", zorder=0)

    ymax = max(expired_values) if expired_values else 0
    text_offset = ymax * 0.03 if ymax > 0 else 0.01
    ax.set_ylim(0, 1 + text_offset * 5)

    for bar, value in zip(bars, expired_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + text_offset,
            f"{value:.3f}",
            ha="center",
            va="bottom"
        )

for j in range(len(ue_ids), len(axes5)):
    fig5.delaxes(axes5[j])

fig5.tight_layout()
plt.show()