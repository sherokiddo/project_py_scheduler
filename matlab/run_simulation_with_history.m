clc; clear; close all;

%% --- Параметры Сравнения ---
numIterations = 5;         % Число итераций для усреднения
simTime = 50;              % Время симуляции для каждой итерации (сек)
numUEs = 10;               % Количество UE
bandwidth = 20e6;          % Ширина полосы (Гц)
scs = 30e3;                % Поднесущая (Гц)
carrierFrequency = 3.5e9;  % Несущая частота (Гц)
gNBTxPower = 46;           % Мощность gNB (дБм)
ueTxPower = 23;            % Мощность UE (дБм)

% --- Выбор Моделей для Сравнения ---
mobilityModel = 'GaussMarkov'; % 'RandomWaypoint' или 'GaussMarkov'
channelModel = 'Custom'; % 'Custom', 'LTE_ETU', 'LTE_EVA', 'LTE_EPA', '5G_TDL_UMa_NLOS', '5G_TDL_RMa_NLOS'

% --- Параметры TDD ---
tddConfig = [1,1,1,1,1,1,1,0,0,0]; % Пример: 7 DL, 3 UL
slotsPerPattern = length(tddConfig);

% --- Алгоритмы для Сравнения ---
schedulerTypes = {'RoundRobin', 'ProportionalFair', 'MaxThroughput','CatBoost'}; %, 'CatBoost'};
% Добавить 'CatBoost', если ML модель обучена и готова

numSchedulers = length(schedulerTypes);
avgSystemThroughput = zeros(numIterations, numSchedulers); % Хранение средней ПРОПУСКНОЙ СПОСОБНОСТИ СИСТЕМЫ
avgUEThroughput = zeros(numIterations, numSchedulers, numUEs); % Хранение средней пропускной способности КАЖДОГО UE
jainFairnessIndex = zeros(numIterations, numSchedulers); % Индекс справедливости Джейна

%% Основной цикл сравнения алгоритмов
fprintf('=== Запуск Сравнения Планировщиков ===\n');
fprintf('Параметры: %d Итераций, %d сек/итер, %d UE, Mob=%s, Chan=%s\n', ...
        numIterations, simTime, numUEs, mobilityModel, channelModel);

for iter = 1:numIterations
    fprintf('\n--- Итерация %d / %d ---\n', iter, numIterations);

    % Создаем один симулятор для всех планировщиков в этой итерации
    % (Чтобы они работали в одинаковых условиях мобильности/канала)
    rng(iter);
    fprintf(' Инициализация симулятора для итерации %d...\n', iter);
    simulator = SystemLevelSimulator(numUEs, bandwidth, scs, simTime, carrierFrequency, mobilityModel, channelModel, gNBTxPower, ueTxPower);

    for i = 1:numSchedulers
        schedulerType = schedulerTypes{i};
        fprintf('  Запуск симуляции для: %s\n', schedulerType);

        % Создаем планировщик
        scheduler = CustomScheduler(simulator.NumRBs, numUEs, schedulerType);

        rng(iter);
        simulator_run = SystemLevelSimulator(numUEs, bandwidth, scs, simTime, carrierFrequency, mobilityModel, channelModel, gNBTxPower, ueTxPower);

        % Запускаем симуляцию без визуализации
        results = runSimulationLocal(scheduler, simulator_run, simTime, tddConfig, slotsPerPattern);

        % Сохраняем результаты
        avgSystemThroughput(iter, i) = results.AvgSystemThroughput;
        avgUEThroughput(iter, i, :) = results.AvgUEThroughput;
        jainFairnessIndex(iter, i) = results.JainFairness;

        fprintf('  Результат %s: Пропускная способность=%.2f, Справедливость=%.3f\n', ...
                schedulerType, results.AvgSystemThroughput, results.JainFairness);
    end % Конец цикла по планировщикам
end % Конец цикла по итерациям

fprintf('\n=== Сравнение Завершено ===\n');

%% Анализ и Построение Графиков

fprintf('Построение итоговых графиков...\n');
iterationsVector = 1:numIterations; % Вектор для оси X

% --- 1. Динамика Средней Пропускной Способности Системы по Итерациям ---
hFigSysThrDyn = figure('Name', 'Сравнение: Динамика Пропускной Способности Системы');
plot(iterationsVector, avgSystemThroughput, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('Номер итерации');
ylabel('Средняя Пропускная Способность Системы');
title(sprintf('Динамика пропускной способности (%d UE, Chan: %s)', numUEs, channelModel));
legend(schedulerTypes, 'Location', 'best');
grid on;
box off;
% Дополнительно можно вывести среднее значение пунктиром
hold on;
meanSystemThroughput = mean(avgSystemThroughput, 1);
plot(iterationsVector, repmat(meanSystemThroughput', 1, numIterations)', '--', 'LineWidth', 1); % Линии для средних
% Добавляем текст со средними значениями в легенду
legend_entries = schedulerTypes;
for i = 1:numSchedulers
    legend_entries{end+1} = sprintf('%s (Avg: %.3f)', schedulerTypes{i}, meanSystemThroughput(i));
end
legend(legend_entries(numSchedulers+1:end), 'Location', 'best', 'AutoUpdate', 'off'); % Отображаем только средние в легенде
hold off;


% --- 2. Динамика Индекса Справедливости Джейна по Итерациям ---
hFigJFIDyn = figure('Name', 'Сравнение: Динамика Индекса Справедливости Джейна');
plot(iterationsVector, jainFairnessIndex, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('Номер итерации');
ylabel('Индекс Справедливости Джейна');
title(sprintf('Динамика справедливости (%d UE, Chan: %s)', numUEs, channelModel));
legend(schedulerTypes, 'Location', 'best');
ylim([0 1.05]); % Индекс Джейна от 0 до 1
grid on;
box off;
% Дополнительно можно вывести среднее значение пунктиром
hold on;
meanJFI = mean(jainFairnessIndex, 1);
plot(iterationsVector, repmat(meanJFI', 1, numIterations)', '--', 'LineWidth', 1); % Линии для средних
% Добавляем текст со средними значениями в легенду
legend_entries_jfi = schedulerTypes;
for i = 1:numSchedulers
     legend_entries_jfi{end+1} = sprintf('%s (Avg: %.3f)', schedulerTypes{i}, meanJFI(i));
end
legend(legend_entries_jfi(numSchedulers+1:end), 'Location', 'best', 'AutoUpdate', 'off');
hold off;


% --- 3. Распределение пропускной способности по UE (Box Plot - ОСТАВЛЯЕМ) ---
hFigUEThr = figure('Name', 'Сравнение: Распределение Пропускной Способности по UE');
ueThrDataForBoxplot = zeros(numIterations * numUEs, numSchedulers);
for i = 1:numSchedulers
    ueThrDataForBoxplot(:, i) = reshape(avgUEThroughput(:, i, :), [], 1);
end
boxplot(ueThrDataForBoxplot, 'Labels', schedulerTypes, 'Whisker', 1.5);
ylabel('Средняя Пропускная Способность UE');
title(sprintf('Распределение пропускной способности по UE (%d итераций, Chan: %s)', numIterations, channelModel));
grid on;
box off;

fprintf('Построение графиков завершено.\n');


%% Локальная функция runSimulation (модифицирована для возврата структуры)
function results = runSimulationLocal(scheduler, simulator, simTime, tddConfig, slotsPerPattern)
    numUEs = scheduler.NumUEs;
    slotsPerPattern = length(tddConfig);
    % Массив для накопления *реализованной* пропускной способности UE в каждом слоте
    slotThroughput = zeros(numUEs, simTime);

    for t = 1:simTime
        % Обновляем сеть и канал
        simulator.updateNetwork(t, scheduler); % Передаем планировщик

        % Обновляем канал в планировщике
        scheduler.updateChannelInfo(simulator.ChannelInfo);

        % Определяем тип слота
        slotType = tddConfig(mod(t-1, slotsPerPattern) + 1);

        % Выполняем планирование
        if slotType == 1 % DL
            scheduler.scheduleDL(t);
            grid = scheduler.getCurrentDLAllocation();
            % Проверяем наличие поля CQI перед использованием
            if isfield(scheduler.ChannelInfo, 'CQI_DL_PerRB') && ~isempty(scheduler.ChannelInfo.CQI_DL_PerRB)
                cqiPerRB = scheduler.ChannelInfo.CQI_DL_PerRB;
            else
                cqiPerRB = ones(numUEs, scheduler.NumRBs); % Заглушка, если CQI нет
                warning('Отсутствует CQI_DL_PerRB в слоте %d для расчета пропускной способности DL.', t);
            end
        else % UL
            scheduler.scheduleUL(t);
            grid = scheduler.getCurrentULAllocation();
             if isfield(scheduler.ChannelInfo, 'CQI_UL_PerRB') && ~isempty(scheduler.ChannelInfo.CQI_UL_PerRB)
                 cqiPerRB = scheduler.ChannelInfo.CQI_UL_PerRB;
             else
                 cqiPerRB = ones(numUEs, scheduler.NumRBs); % Заглушка
                 warning('Отсутствует CQI_UL_PerRB в слоте %d для расчета пропускной способности UL.', t);
             end
        end

        % Рассчитываем и сохраняем *реализованную* пропускную способность в этом слоте
        currentSlotRates = zeros(numUEs, 1);
         for ue = 1:numUEs
             allocatedRBs = find(grid == ue);
             if ~isempty(allocatedRBs)
                 rateUE = 0;
                 for rb_idx = 1:length(allocatedRBs)
                     rb = allocatedRBs(rb_idx);
                     % Проверка границ CQI матрицы
                     if ue <= size(cqiPerRB, 1) && rb <= size(cqiPerRB, 2)
                         mcs = scheduler.getMCS(cqiPerRB(ue, rb));
                         rateUE = rateUE + mcs.efficiency; % Суммируем эффективность
                     else
                         warning('Индекс CQI вне диапазона в runSimulationLocal: ue=%d, rb=%d', ue, rb);
                     end
                 end
                 currentSlotRates(ue) = rateUE;
             end
         end
         % Защита от NaN или Inf, если были ошибки
         currentSlotRates(isnan(currentSlotRates) | isinf(currentSlotRates)) = 0;
         slotThroughput(:, t) = currentSlotRates;
         

         % Обновляем внутреннюю AvgThroughput планировщика (для PF)
         scheduler.updateAverageThroughput(ternary(slotType==1,'DL','UL'));

         % Обновляем PreviousAllocation в планировщике (важно!)
         if slotType == 1
              scheduler.PreviousDLAllocation = scheduler.DL_Grid;
         else
              scheduler.PreviousULAllocation = scheduler.UL_Grid;
         end

    end % Конец цикла по времени t

    % Расчет итоговых метрик
    avgUEThroughputFinal = mean(slotThroughput, 2); % Среднее по времени для каждого UE [NumUEs x 1]
    avgSystemThroughputFinal = sum(avgUEThroughputFinal); % Суммарная средняя по системе

    % Индекс справедливости Джейна
    sumUEThr = sum(avgUEThroughputFinal);
    sumSqUEThr = sum(avgUEThroughputFinal.^2);
    if sumUEThr > 1e-9 % Избегаем деления на ноль
        jainFairness = (sumUEThr^2) / (numUEs * sumSqUEThr);
    else
        jainFairness = 0; % Если пропускная способность нулевая
    end
    % Ограничиваем сверху 1 (на случай очень малых значений)
    jainFairness = min(jainFairness, 1);

    % Возвращаем результаты в структуре
    results = struct(...
        'AvgSystemThroughput', avgSystemThroughputFinal, ...
        'AvgUEThroughput', avgUEThroughputFinal', ... % Транспонируем для согласования размерности
        'JainFairness', jainFairness ...
    );
end

% Простая тернарная функция (если нет Statistics and ML Toolbox)
function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end