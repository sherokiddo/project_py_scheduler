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
        
        SHOW_TITLES = False

        # ===== ГРАФИКИ ПРОПУСКНОЙ СПОСОБНОСТИ =====
        # График пропускной способности соты
        plt.figure(figsize=(10, 6))
        plt.plot(tti_range, metrics["cell_throughput"])
        if SHOW_TITLES:
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
        if SHOW_TITLES:    
            plt.suptitle(f"Пропускная способность каждого пользователя при планировщике {scheduler}", fontsize=14)
        plt.subplots_adjust(top=0.92)
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
        if SHOW_TITLES:
            plt.title(f"Средняя пропускная способность каждого пользователя при планировщике {scheduler}")
        plt.ylabel("Пропускная способность (Мбит/с)")
        plt.grid(True, zorder=0)
        plt.tight_layout()
        plt.show()
        
        # ===== ГРАФИКИ ИНДЕКСА СПРАВЕДЛИВОСТИ =====  
        # График индекса справедливости во времени
        plt.figure(figsize=(10, 6))
        plt.plot(tti_range, metrics["jain_index_per_frame"], color='green')
        if SHOW_TITLES:
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
    
    if SHOW_TITLES:
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

    if SHOW_TITLES:
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

    if SHOW_TITLES:
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
    
    if SHOW_TITLES:
        plt.title("CDF спектральной эффективности")
    plt.xlabel("Спектральная эффективность (бит/с/Гц)")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    boxplot_data = []
    xtick_labels = []
    colors = []
    
    # График сравнения Boxplot'ов всех планировщиков
    for scheduler, metrics in metrics_data.items():
        user_throughput = metrics["user_throughput"]
        scheduler_short = scheduler_labels.get(scheduler, scheduler)
        color = scheduler_colors.get(scheduler, "gray")
    
        for uid in sorted(user_throughput.keys(), key=int):
            boxplot_data.append(user_throughput[uid])
            xtick_labels.append(f"UE{uid} ({scheduler_short})")
            colors.append(color)
    
    bp = plt.boxplot(boxplot_data,
                     patch_artist=True,
                     labels=xtick_labels,
                     widths=0.3,
                     medianprops=dict(color='orange', linewidth=2))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.xticks(rotation=30)
    plt.ylabel("Пропускная способность (Мбит/с)")
    if SHOW_TITLES:
        plt.title("Boxplot пропускной способности пользователей (все планировщики)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_scheduler_efficiency_from_file(filename='scheduler_efficiency.json'):
    with open(filename, 'r') as f:
        all_results = json.load(f)

    for scheduler_name, experiments in all_results.items():
        
        plt.rcParams.update({'font.size': 12})
        
        # График среднего затраченного времени на планирование ресурсов в одном TTI
        x_users = []
        y_mean_times = []
        
        for experiment in experiments:
            num_users = experiment["num_users"]
            mean_times = experiment["mean_elapsed_time_array"]
            if not mean_times:
                continue
            mean_time = np.mean(mean_times)
        
            x_users.append(num_users)
            y_mean_times.append(mean_time)
        
        sorted_pairs = sorted(zip(x_users, y_mean_times))
        x_users_sorted, y_mean_times_sorted = zip(*sorted_pairs)
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Среднее время распределения ресурсов в одном TTI\nПланировщик: {scheduler_name}")
        plt.xlabel("Кол-во пользователей")
        plt.ylabel("Среднее время (мс)")
        plt.plot(x_users_sorted, y_mean_times_sorted, marker='o', linestyle='-')
        plt.xticks(ticks=range(min(x_users_sorted), max(x_users_sorted) + 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # График CDF времени распределения ресурсов в одном TTI
        plt.figure(figsize=(10, 6))
        plt.title(f"CDF времени распределения ресурсов в одном TTI\nПланировщик: {scheduler_name}")
        plt.xlabel("Время (мс)")
        plt.ylabel("CDF")

        for experiment in experiments:
            num_users = experiment["num_users"]
            elapsed_time_array = experiment["elapsed_time_array"]

            if not elapsed_time_array:
                continue

            ecdf = ECDF(elapsed_time_array)
            plt.step(ecdf.x, ecdf.y, label=f"{num_users} UEs")

        plt.legend(title="Кол-во пользователей")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # График Boxplot’ов времени распределения ресурсов в одном TTI
        plt.figure(figsize=(10, 6))
        plt.title(f"Boxplot времени распределения ресурсов в одном TTI\nПланировщик: {scheduler_name}")
        plt.xlabel("Кол-во пользователей")
        plt.ylabel("Время (мс)")

        # Сортировка по возрастанию пользователей для читаемости
        experiments_sorted = sorted(experiments, key=lambda x: x["num_users"])
        boxplot_data = [exp["elapsed_time_array"] for exp in experiments_sorted]
        labels = [f"{exp['num_users']} UEs" for exp in experiments_sorted]

        plt.boxplot(boxplot_data, 
                    labels=labels, 
                    widths=0.3,
                    patch_artist=True,
                    boxprops=dict(facecolor='blue'),
                    medianprops=dict(color='orange', linewidth=2))

        plt.grid(True)
        plt.tight_layout()
        plt.show()

    
if __name__ == "__main__":
    plot_scheduler_metrics_from_file()
    #plot_scheduler_efficiency_from_file()