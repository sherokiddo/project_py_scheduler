import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

def plot_scheduler_metrics_from_file(json_file='metrics_results.json'):
    if not os.path.exists(json_file):
        print(f"[Ошибка] Файл {json_file} не найден.")
        return

    with open(json_file, 'r') as f:
        metrics_data = json.load(f)

    if not metrics_data:
        print("[Ошибка] Файл пуст или содержит некорректные данные.")
        return

    for scheduler, metrics in metrics_data.items():
        tti_range = range(0, metrics["sim_duration"], 10)
        plt.rcParams.update({'font.size': 12})

        # ===== ГРАФИКИ ПРОПУСКНОЙ СПОСОБНОСТИ =====
        # График пропускной способности соты
        plt.figure(figsize=(10, 6))
        plt.plot(tti_range, metrics["cell_throughput"])
        plt.title(f"Пропускная способность соты при планировщике {scheduler}")
        plt.xlabel("Время (мс)")
        plt.ylabel("Пропускная способность (Мбит/с)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # График пропускной способности для каждого пользователя
        user_throughput = metrics["user_throughput"]
        user_ids = sorted(user_throughput.keys(), key=int)
        num_users = len(user_ids)
        
        cols = 2
        rows = math.ceil(num_users / cols)   
        plt.figure(figsize=(10, 6))
        for idx, uid in enumerate(user_ids):
            plt.subplot(rows, cols, idx + 1)
            plt.plot(tti_range, user_throughput[uid], color='red')
            plt.title(f"Пропускная способность UE{uid}")
            plt.xlabel("Время (мс)")
            plt.ylabel("Пропускная способность (Мбит/с)")
            plt.grid(True)
            plt.tight_layout()
        plt.suptitle(f"Пропускная способность каждого пользователя при планировщике {scheduler}", fontsize=14)
        plt.subplots_adjust(top=0.92)
        plt.show()

        # Boxplot'ы для пропускной способности пользователей
        user_throughput = metrics["user_throughput"]
        user_ids = sorted(user_throughput.keys(), key=int)
        user_data = [user_throughput[uid] for uid in user_ids]

        plt.figure(figsize=(10, 6))
        plt.boxplot(user_data, labels=[f"UE{uid}" for uid in user_ids],
                    patch_artist=True, medianprops=dict(color='orange', linewidth=2))
        plt.title(f"Boxplot пропускной способности пользователей при планировщике {scheduler}")
        plt.ylabel("Пропускная способность (Мбит/с)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Столбчатый график средней пропускной способности для каждого пользователя
        avg_user_throughput = metrics["avg_user_throughput"]
        ue_labels = [f"UE{uid}" for uid in avg_user_throughput]
        avg_values = list(avg_user_throughput.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(ue_labels, avg_values, edgecolor='black', width=0.3, zorder=3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.2f}",
                     ha='center', va='bottom', fontsize=10)
            
        plt.title(f"Средняя пропускная способность каждого пользователя при планировщике {scheduler}")
        plt.ylabel("Пропускная способность (Мбит/с)")
        plt.grid(True, zorder=0)
        plt.tight_layout()
        plt.show()
        
        # ===== ГРАФИКИ ИНДЕКСА СПРАВЕДЛИВОСТИ =====  
        # График индекса справедливости во времени
        plt.figure(figsize=(10, 6))
        plt.plot(tti_range, metrics["jain_index_per_frame"], color='green')
        plt.title(f"Индекс справедливости Джайна во времени при планировщике {scheduler}")
        plt.xlabel("Время (мс)")
        plt.ylabel("Справедливость")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # ===== ГРАФИКИ СРАВНЕНИЯ ВСЕХ ПЛАНИРОВЩИКОВ =====
    scheduler_colors = {
        "RoundRobinScheduler": "green",
        "ProportionalFairScheduler": "blue",
        "BestCQIScheduler": "red"
    }    
    
    scheduler_labels = {
        "RoundRobinScheduler": "RR",
        "ProportionalFairScheduler": "PF",
        "BestCQIScheduler": "BCQI"
    }
    
    # График сравнения пропускной способности для всех планировщиков 
    plt.figure(figsize=(10, 6))
    for scheduler, metrics in metrics_data.items():
        tti_range = range(0, metrics["sim_duration"], 10)
        color = scheduler_colors.get(scheduler, "gray")

        plt.plot(tti_range, metrics["cell_throughput"],
                 label=scheduler_labels.get(scheduler, scheduler),
                 color=color)

    plt.title("Сравнение пропускной способности соты для разных планировщиков")
    plt.xlabel("Время (мс)")
    plt.ylabel("Пропускная способность (Мбит/с)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # График сравнения справедливости всех планировщиков
    schedulers = list(metrics_data.keys())
    fairness_values = [metrics_data[s]["jain_index_overall"] for s in schedulers]

    plt.figure(figsize=(6, 6))
    bar_colors = [scheduler_colors.get(s, "gray") for s in schedulers]
    x_labels = [scheduler_labels.get(s, s) for s in schedulers]
    bars = plt.bar(x_labels,
                   fairness_values,
                   color=bar_colors,
                   edgecolor='black',
                   width=0.7,
                   zorder=3)

    for bar, val in zip(bars, fairness_values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}",
                 ha='center', va='bottom', fontsize=12)

    plt.title("Индекс справедливости Джайна для разных планировщиков")
    plt.ylabel("Справедливость")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', zorder=0)
    plt.tight_layout()
    plt.show()
    
    # График сравнения спектральной эффективности всех планировщиков
    plt.figure(figsize=(10, 6))

    for scheduler, metrics in metrics_data.items():
        tti_range = range(0, metrics["sim_duration"], 10)
        color = scheduler_colors.get(scheduler, "gray")

        plt.plot(tti_range, metrics["spectral_efficiency"],
                 label=scheduler_labels.get(scheduler, scheduler),
                 color=color)

    plt.title("Сравнение спектральной эффективности разных планировщиков")
    plt.xlabel("Время (мс)")
    plt.ylabel("Спектральная эффективность (бит/с/Гц)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # График CDF для спектральной эффективности
    plt.figure(figsize=(6, 6))
    
    for scheduler, metrics in metrics_data.items():
        spectral_eff = np.array(metrics["spectral_efficiency"])
        ecdf = ECDF(spectral_eff)
        color = scheduler_colors.get(scheduler, "gray")

        plt.step(ecdf.x, ecdf.y, where='post', color=color, label=scheduler_labels.get(scheduler, scheduler))
        
    plt.title("CDF спектральной эффективности")
    plt.xlabel("Спектральная эффективность (бит/с/Гц)")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    plot_scheduler_metrics_from_file()