% Файл: CustomScheduler.m
% --- ПОЛНЫЙ ИСПРАВЛЕННЫЙ КОД (с корректной проверкой isfield) ---
classdef CustomScheduler < handle
    properties
        % Конфигурация
        NumRBs
        NumUEs
        SchedulerType
        SymbolsPerSlot = 14

        % Текущее состояние канала
        ChannelInfo = struct();

        % Состояние планировщика
        AvgThroughput
        DL_Grid         % [1 x NumRBs], ID UE или 0
        UL_Grid         % [1 x NumRBs], ID UE или 0
        PreviousDLAllocation % [1 x NumRBs] из t-1
        PreviousULAllocation % [1 x NumRBs] из t-1

        % ML Параметры
        mlEnabled = false;
        collectingMLData = false;
        mlTrainingTargetScheduler = 'ProportionalFair';
        trainingDataRB = []; % Данные для RB модели [Feat1,...,FeatN, TargetUE_ID]
        predictionCache = containers.Map('KeyType','char','ValueType','any');

        % HARQ (Не используется)
        HARQ_Processes

        % Логгирование
        enableLogging = true;
    end

    methods
        % =================================================================
        % --- Конструктор ---
        % =================================================================
        function obj = CustomScheduler(numRBs, numUEs, type)
             obj.NumRBs = numRBs;
             obj.NumUEs = numUEs;
             obj.SchedulerType = type;
             obj.AvgThroughput = ones(numUEs, 1);
             obj.DL_Grid = zeros(1, numRBs);
             obj.UL_Grid = zeros(1, numRBs);
             obj.PreviousDLAllocation = zeros(1, numRBs);
             obj.PreviousULAllocation = zeros(1, numRBs);
             obj.HARQ_Processes = struct('UE', cell(8,1), 'RB', cell(8,1), 'Status', 'ACK');
             if strcmpi(obj.SchedulerType, 'CatBoost')
                obj.mlEnabled = true; % Включаем ML, если выбран CatBoost
                fprintf('Планировщик инициализирован: Тип=%s (ML Enabled, Predicts UE per RB)\n', type);
             else
                 fprintf('Планировщик инициализирован: Тип=%s\n', type);
             end
        end

        % =================================================================
        % --- Обновление / Получение Данных ---
        % =================================================================
        function updateChannelInfo(obj, currentChannelInfo)
            obj.ChannelInfo = currentChannelInfo;
        end

        function alloc = getCurrentDLAllocation(obj)
            alloc = obj.DL_Grid;
        end
        function alloc = getCurrentULAllocation(obj)
            alloc = obj.UL_Grid;
        end

        % =================================================================
        % --- Основные Методы Планирования ---
        % =================================================================
        function scheduleDL(obj, t)
             % --- КОРРЕКТНАЯ ПРОВЕРКА ChannelInfo ---
             if isempty(fieldnames(obj.ChannelInfo)) || ~isfield(obj.ChannelInfo, 'CQI_DL_PerRB') || isempty(obj.ChannelInfo.CQI_DL_PerRB)
                 warning('ChannelInfo DL недоступен в слоте %d. Пропуск.', t);
                 obj.DL_Grid = zeros(1, obj.NumRBs); % Сброс
                 obj.PreviousDLAllocation = obj.DL_Grid; % Обновляем предыдущий
                 return;
             end
             % --- Конец КОРРЕКТНОЙ ПРОВЕРКИ ---

             obj.DL_Grid = zeros(1, obj.NumRBs); % Обнуляем текущую сетку

             if obj.collectingMLData % РЕЖИМ СБОРА ДАННЫХ
                  targetGrid = obj.runTeacherScheduler('DL', t);
                  obj.collectTrainingDataRB('DL', targetGrid);
                  obj.DL_Grid = targetGrid; % Используем решение учителя
                  if mod(t,50)==0, fprintf('.'); end
             else % ОБЫЧНЫЙ РЕЖИМ / ПРЕДСКАЗАНИЕ
                  switch obj.SchedulerType
                      case 'RoundRobin', obj.roundRobinDL(t);
                      case 'ProportionalFair', obj.proportionalFairDL();
                      case 'MaxThroughput', obj.maxThroughputDL();
                      case 'CatBoost'
                          if obj.mlEnabled
                              obj.catBoostSchedulerDL();
                          else
                               warning('CatBoost выбран, но ML отключен (mlEnabled=false). Используется RoundRobin.');
                               obj.roundRobinDL(t);
                          end
                      otherwise
                          warning('Неизвестный SchedulerType: %s. Используется RoundRobin.', obj.SchedulerType);
                          obj.roundRobinDL(t);
                  end
             end

             obj.updateAverageThroughput('DL'); % Обновляем среднюю пропускную способность
             if obj.enableLogging && ~obj.collectingMLData % Вызываем логирование, если нужно
                  obj.logSchedule('DL', t);
             end
             obj.PreviousDLAllocation = obj.DL_Grid; % Сохраняем для следующего шага
        end

         function scheduleUL(obj, t)
             % --- КОРРЕКТНАЯ ПРОВЕРКА ChannelInfo ---
             if isempty(fieldnames(obj.ChannelInfo)) || ~isfield(obj.ChannelInfo,'CQI_UL_PerRB') || isempty(obj.ChannelInfo.CQI_UL_PerRB)
                 warning('ChannelInfo UL недоступен в слоте %d. Пропуск.', t);
                 obj.UL_Grid = zeros(1, obj.NumRBs); % Сброс
                 obj.PreviousULAllocation = obj.UL_Grid; % Обновляем предыдущий
                 return;
             end
             % --- Конец КОРРЕКТНОЙ ПРОВЕРКИ ---

             obj.UL_Grid = zeros(1, obj.NumRBs); % Обнуляем текущую сетку

              if obj.collectingMLData % РЕЖИМ СБОРА ДАННЫХ
                   targetGrid = obj.runTeacherScheduler('UL', t);
                   obj.collectTrainingDataRB('UL', targetGrid);
                   obj.UL_Grid = targetGrid; % Используем решение учителя
                   if mod(t,50)==0, fprintf('.'); end
              else % ОБЫЧНЫЙ РЕЖИМ / ПРЕДСКАЗАНИЕ
                   switch obj.SchedulerType
                      case 'RoundRobin', obj.roundRobinUL(t);
                      case 'ProportionalFair', obj.proportionalFairUL();
                      case 'MaxThroughput', obj.maxThroughputUL();
                      case 'CatBoost'
                          if obj.mlEnabled
                              obj.catBoostSchedulerUL();
                          else
                               warning('CatBoost выбран, но ML отключен (mlEnabled=false). Используется RoundRobin.');
                               obj.roundRobinUL(t);
                          end
                      otherwise
                          warning('Неизвестный SchedulerType: %s. Используется RoundRobin.', obj.SchedulerType);
                          obj.roundRobinUL(t);
                  end
              end

             obj.updateAverageThroughput('UL'); % Обновляем среднюю пропускную способность
             if obj.enableLogging && ~obj.collectingMLData % Вызываем логирование, если нужно
                  obj.logSchedule('UL', t);
             end
             obj.PreviousULAllocation = obj.UL_Grid; % Сохраняем для следующего шага
         end

        % =================================================================
        % --- Алгоритмы и Вспомогательные Функции ---
        % =================================================================

        % --- Запуск планировщика-учителя (для сбора данных) ---
        function targetGrid = runTeacherScheduler(obj, direction, t)
             originalDLGrid = obj.DL_Grid; originalULGrid = obj.UL_Grid;
             try
                 switch lower(obj.mlTrainingTargetScheduler)
                     case 'proportionalfair'
                         if strcmpi(direction, 'DL'), obj.proportionalFairDL(); else, obj.proportionalFairUL(); end
                     case 'maxthroughput'
                          if strcmpi(direction, 'DL'), obj.maxThroughputDL(); else, obj.maxThroughputUL(); end
                     case 'roundrobin'
                          if strcmpi(direction, 'DL'), obj.roundRobinDL(t); else, obj.roundRobinUL(t); end
                     otherwise
                          warning('Неизвестный mlTrainingTargetScheduler: %s. Исп. ProportionalFair.', obj.mlTrainingTargetScheduler);
                          if strcmpi(direction, 'DL'), obj.proportionalFairDL(); else, obj.proportionalFairUL(); end
                 end
             catch ME
                  warning('Ошибка планировщика-учителя (%s): %s. Сбор данных пропущен.', obj.mlTrainingTargetScheduler, ME.message);
                   if strcmpi(direction, 'DL'), targetGrid = originalDLGrid; else, targetGrid = originalULGrid; end
                   obj.DL_Grid = originalDLGrid; obj.UL_Grid = originalULGrid; return; % Возвращаем исходную сетку при ошибке
             end
             if strcmpi(direction, 'DL'), targetGrid = obj.DL_Grid; else, targetGrid = obj.UL_Grid; end
             obj.DL_Grid = originalDLGrid; obj.UL_Grid = originalULGrid; % Восстанавливаем
        end

        % --- Алгоритмы Планирования: RoundRobin ---
        function roundRobinDL(obj, t)
            startUE = mod(t - 1, obj.NumUEs) + 1;
            rbsPerUE = floor(obj.NumRBs / obj.NumUEs);
            currentRB = 1;
            tempGrid = zeros(1, obj.NumRBs); % Работаем с временной сеткой
            for i = 1:obj.NumUEs
                ueIdx = mod(startUE + i - 2, obj.NumUEs) + 1;
                rbEnd = min(currentRB + rbsPerUE - 1, obj.NumRBs);
                if currentRB <= rbEnd
                    tempGrid(currentRB : rbEnd) = ueIdx;
                    currentRB = rbEnd + 1;
                end
                 if i == obj.NumUEs && currentRB <= obj.NumRBs
                     tempGrid(currentRB : obj.NumRBs) = ueIdx;
                 end
            end
            obj.DL_Grid = tempGrid; % Записываем результат
        end
        function roundRobinUL(obj, t)
             startUE = mod(t - 1, obj.NumUEs) + 1;
            rbsPerUE = floor(obj.NumRBs / obj.NumUEs);
            currentRB = 1;
            tempGrid = zeros(1, obj.NumRBs);
            for i = 1:obj.NumUEs
                ueIdx = mod(startUE + i - 2, obj.NumUEs) + 1;
                rbEnd = min(currentRB + rbsPerUE - 1, obj.NumRBs);
                 if currentRB <= rbEnd
                     tempGrid(currentRB : rbEnd) = ueIdx;
                     currentRB = rbEnd + 1;
                 end
                 if i == obj.NumUEs && currentRB <= obj.NumRBs
                     tempGrid(currentRB : obj.NumRBs) = ueIdx;
                 end
            end
            obj.UL_Grid = tempGrid; % Записываем результат
        end

        % --- Алгоритмы Планирования: ProportionalFair ---
        function proportionalFairDL(obj)
            if isempty(fieldnames(obj.ChannelInfo)) || ~isfield(obj.ChannelInfo, 'CQI_DL_PerRB') || isempty(obj.ChannelInfo.CQI_DL_PerRB)
                warning('PF DL: ChannelInfo недоступен. Пропуск.'); obj.DL_Grid = zeros(1, obj.NumRBs); return;
            end
            cqiPerRB = obj.ChannelInfo.CQI_DL_PerRB; avgThr = obj.AvgThroughput;
            avgThr(avgThr < 1e-9) = 1e-9; metrics = zeros(obj.NumUEs, obj.NumRBs);
            for ue = 1:obj.NumUEs, for rb = 1:obj.NumRBs, mcs = obj.getMCS(cqiPerRB(ue, rb)); metrics(ue, rb) = mcs.efficiency / avgThr(ue); end, end
            obj.greedyRBAllocation(metrics, 'DL');
        end
        function proportionalFairUL(obj)
             if isempty(fieldnames(obj.ChannelInfo)) || ~isfield(obj.ChannelInfo, 'CQI_UL_PerRB') || isempty(obj.ChannelInfo.CQI_UL_PerRB)
                warning('PF UL: ChannelInfo недоступен. Пропуск.'); obj.UL_Grid = zeros(1, obj.NumRBs); return;
             end
            cqiPerRB = obj.ChannelInfo.CQI_UL_PerRB; avgThr = obj.AvgThroughput;
            avgThr(avgThr < 1e-9) = 1e-9; metrics = zeros(obj.NumUEs, obj.NumRBs);
             for ue = 1:obj.NumUEs, for rb = 1:obj.NumRBs, mcs = obj.getMCS(cqiPerRB(ue, rb)); metrics(ue, rb) = mcs.efficiency / avgThr(ue); end, end
            obj.greedyRBAllocation(metrics, 'UL');
        end

        % --- Алгоритмы Планирования: MaxThroughput ---
         function maxThroughputDL(obj)
              if isempty(fieldnames(obj.ChannelInfo)) || ~isfield(obj.ChannelInfo, 'CQI_DL_PerRB') || isempty(obj.ChannelInfo.CQI_DL_PerRB)
                warning('MT DL: ChannelInfo недоступен. Пропуск.'); obj.DL_Grid = zeros(1, obj.NumRBs); return;
             end
             cqiPerRB = obj.ChannelInfo.CQI_DL_PerRB; metrics = zeros(obj.NumUEs, obj.NumRBs);
             for ue = 1:obj.NumUEs, for rb = 1:obj.NumRBs, mcs = obj.getMCS(cqiPerRB(ue, rb)); metrics(ue, rb) = mcs.efficiency; end, end
             obj.greedyRBAllocation(metrics, 'DL');
         end
        function maxThroughputUL(obj)
             if isempty(fieldnames(obj.ChannelInfo)) || ~isfield(obj.ChannelInfo, 'CQI_UL_PerRB') || isempty(obj.ChannelInfo.CQI_UL_PerRB)
                warning('MT UL: ChannelInfo недоступен. Пропуск.'); obj.UL_Grid = zeros(1, obj.NumRBs); return;
             end
            cqiPerRB = obj.ChannelInfo.CQI_UL_PerRB; metrics = zeros(obj.NumUEs, obj.NumRBs);
             for ue = 1:obj.NumUEs, for rb = 1:obj.NumRBs, mcs = obj.getMCS(cqiPerRB(ue, rb)); metrics(ue, rb) = mcs.efficiency; end, end
             obj.greedyRBAllocation(metrics, 'UL');
        end

        % --- Общий Жадный Алгоритм Распределения RB ---
        function greedyRBAllocation(obj, metrics, direction)
              availableRB_Mask = true(1, obj.NumRBs);
             numAvailableRB = obj.NumRBs;
             targetGrid = zeros(1, obj.NumRBs);
             while numAvailableRB > 0
                 bestMetric = -Inf; bestUE = -1; bestRB = -1;
                 nonInfMetrics = metrics > -Inf;
                 if ~any(nonInfMetrics(:)), break; end
                 validMetrics = metrics;
                 validMetrics(~nonInfMetrics) = -Inf;
                 validMetrics(:, ~availableRB_Mask) = -Inf;
                 [maxVal, linearIdx] = max(validMetrics(:));
                 if isempty(maxVal) || maxVal <= -Inf, break; end
                 [bestUE, bestRB] = ind2sub(size(metrics), linearIdx);
                 targetGrid(bestRB) = bestUE;
                 availableRB_Mask(bestRB) = false;
                 numAvailableRB = numAvailableRB - 1;
                 metrics(:, bestRB) = -Inf;
             end
              if strcmpi(direction, 'DL'), obj.DL_Grid = targetGrid; else, obj.UL_Grid = targetGrid; end
        end

        % --- Обновление Средней Пропускной Способности ---
         function updateAverageThroughput(obj, direction)
              currentRates = zeros(obj.NumUEs, 1);
             if strcmpi(direction, 'DL')
                 grid = obj.DL_Grid;
                 if isfield(obj.ChannelInfo, 'CQI_DL_PerRB') && ~isempty(obj.ChannelInfo.CQI_DL_PerRB), cqiPerRB = obj.ChannelInfo.CQI_DL_PerRB; else, cqiPerRB = ones(obj.NumUEs, obj.NumRBs); end
             else
                 grid = obj.UL_Grid;
                  if isfield(obj.ChannelInfo, 'CQI_UL_PerRB') && ~isempty(obj.ChannelInfo.CQI_UL_PerRB), cqiPerRB = obj.ChannelInfo.CQI_UL_PerRB; else, cqiPerRB = ones(obj.NumUEs, obj.NumRBs); end
             end
             for ue = 1:obj.NumUEs
                 allocatedRBs = find(grid == ue);
                 if ~isempty(allocatedRBs)
                     rateUE = 0;
                     for rb_idx = 1:length(allocatedRBs)
                         rb = allocatedRBs(rb_idx);
                          if ue <= size(cqiPerRB, 1) && rb <= size(cqiPerRB, 2)
                             mcs = obj.getMCS(cqiPerRB(ue, rb));
                             rateUE = rateUE + mcs.efficiency;
                          end
                     end
                     currentRates(ue) = rateUE;
                 end
             end
             alpha = 0.1;
             obj.AvgThroughput = (1 - alpha) * obj.AvgThroughput + alpha * currentRates;
         end

        % --- Получение MCS ---
        function mcs = getMCS(~, cqi)
             mcsTable = [ 1, 0.15; 2, 0.23; 3, 0.38; 4, 0.60; 5, 0.88; 6, 1.18; 7, 1.48; 8, 1.91; 9, 2.41; 10, 2.73; 11, 3.32; 12, 3.90; 13, 4.52; 14, 5.12; 15, 5.55];
            cqi = max(1, min(15, round(cqi)));
            mcs.code = cqi;
            mcs.efficiency = mcsTable(cqi, 2);
        end

        % --- Логирование Расписания ---
         function logSchedule(obj, direction, t)
              if strcmp(direction, 'DL')
                 grid = obj.DL_Grid;
                 if isfield(obj.ChannelInfo, 'CQI_DL_PerRB') && ~isempty(obj.ChannelInfo.CQI_DL_PerRB), cqiPerRB = obj.ChannelInfo.CQI_DL_PerRB; else, cqiPerRB = ones(obj.NumUEs, obj.NumRBs); end
                 if isfield(obj.ChannelInfo, 'CQI_DL') && ~isempty(obj.ChannelInfo.CQI_DL), cqiAvg = obj.ChannelInfo.CQI_DL; else, cqiAvg = ones(obj.NumUEs, 1); end
             else
                 grid = obj.UL_Grid;
                 if isfield(obj.ChannelInfo, 'CQI_UL_PerRB') && ~isempty(obj.ChannelInfo.CQI_UL_PerRB), cqiPerRB = obj.ChannelInfo.CQI_UL_PerRB; else, cqiPerRB = ones(obj.NumUEs, obj.NumRBs); end
                 if isfield(obj.ChannelInfo, 'CQI_UL') && ~isempty(obj.ChannelInfo.CQI_UL), cqiAvg = obj.ChannelInfo.CQI_UL; else, cqiAvg = ones(obj.NumUEs, 1); end
             end
             fprintf('\n--- %s Слот %d [%s] Распределение ---\n', direction, t, obj.SchedulerType);
             anyAllocated = false;
             for ue = 1:obj.NumUEs
                 rbList = find(grid == ue);
                 if ~isempty(rbList)
                     anyAllocated = true;
                     valid_rbList = rbList(rbList <= size(cqiPerRB, 2)); % Доп. проверка на всякий случай
                     if ue <= size(cqiPerRB, 1) && ~isempty(valid_rbList)
                         avgCQIonRB = mean(cqiPerRB(ue, valid_rbList));
                         fprintf('UE %d: RB %s | AvgCQI=%.1f (OverallAvg=%d)\n', ue, mat2str(rbList), avgCQIonRB, cqiAvg(ue));
                     else
                          fprintf('UE %d: RB %s | Не удалось получить CQI\n', ue, mat2str(rbList));
                     end
                 end
             end
             if ~anyAllocated, fprintf('  Ресурсы никому не выделены.\n'); end
             fprintf('-------------------------------------\n');
         end

        % =================================================================
        % --- Методы для ML (CatBoost - Вариант Б) ---
        % =================================================================
        function catBoostSchedulerDL(obj)
            decisionsMatrix = obj.predictWithCatboost('DL');
            obj.applyMLDecisions(decisionsMatrix, 'DL');
         end
        function catBoostSchedulerUL(obj)
            decisionsMatrix = obj.predictWithCatboost('UL');
            obj.applyMLDecisions(decisionsMatrix, 'UL');
         end

        function decisionsMatrix = predictWithCatboost(obj, direction)
             featuresPerRB = obj.generateFeaturesPerRB(direction);
             if isempty(featuresPerRB)
                 decisionsMatrix = obj.generateRandomAllocationMatrix(); return;
             end

             % --- Кеширование (Временно отключено) ---
             useCache = false;
             cacheKey = '';
             % if useCache
             %     featureSample = featuresPerRB(1,:);
             %     cacheKey = sprintf('RB_%s_%s', direction, DataHash(featureSample));
             %     if obj.predictionCache.isKey(cacheKey)
             %         decisionsMatrix = obj.predictionCache(cacheKey); fprintf('[Cache Hit] '); return;
             %     end
             % end

             % --- Взаимодействие с Python через Файлы ---
             try writematrix(featuresPerRB, 'current_state.csv');
             catch ME, warning('Ошибка записи current_state.csv: %s. Случ. решения.', ME.message); decisionsMatrix = obj.generateRandomAllocationMatrix(); return; end

             fprintf('Вызов Python RB (%s)... ', direction);
             cmd = sprintf('python catboost_scheduler.py %d %s %d', obj.NumRBs, direction, obj.NumUEs);
             tic;
             [status, cmdout] = system(cmd);
             elapsedTime = toc;
             fprintf('Статус: %d. Время: %.3f сек.\n', status, elapsedTime);

             if status ~= 0 || contains(cmdout, 'ERROR', 'IgnoreCase', true)
                 warning('Ошибка Python RB (%s): %s. Случайные решения.', direction, cmdout);
                 decisionsMatrix = obj.generateRandomAllocationMatrix();
             else
                 try
                     decisionsMatrix = readmatrix('predictions.csv');
                     if ~isequal(size(decisionsMatrix), [obj.NumUEs, obj.NumRBs])
                          warning('Размер предсказаний RB (%dx%d) != ожидаемому (%dx%d). Случ. решения.', size(decisionsMatrix,1), size(decisionsMatrix,2), obj.NumUEs, obj.NumRBs);
                          decisionsMatrix = obj.generateRandomAllocationMatrix();
                     end
                 catch ME
                      warning('Ошибка чтения predictions.csv: %s. Случайные решения.', ME.message);
                      decisionsMatrix = obj.generateRandomAllocationMatrix();
                 end
             end

             if useCache && ~isempty(cacheKey)
                 obj.predictionCache(cacheKey) = decisionsMatrix;
             end
        end

        function featuresPerRB = generateFeaturesPerRB(obj, direction)
              featuresPerRB = [];
             if isempty(fieldnames(obj.ChannelInfo)), warning('generateFeaturesPerRB: ChannelInfo пусто.'); return; end

             if strcmpi(direction,'DL')
                 if ~isfield(obj.ChannelInfo,'CQI_DL_PerRB'), warning('generateFeaturesPerRB: CQI_DL_PerRB отсутствует.'); return; end
                 cqiPerRB = obj.ChannelInfo.CQI_DL_PerRB;
                 if ~isfield(obj.ChannelInfo,'TrafficDL'), traffic = zeros(obj.NumUEs,1); else, traffic = obj.ChannelInfo.TrafficDL; end
                 if ~isfield(obj.ChannelInfo,'TrafficBuffer'), buffer = zeros(obj.NumUEs,1); else, buffer = obj.ChannelInfo.TrafficBuffer; end
             else % UL
                  if ~isfield(obj.ChannelInfo,'CQI_UL_PerRB'), warning('generateFeaturesPerRB: CQI_UL_PerRB отсутствует.'); return; end
                  cqiPerRB = obj.ChannelInfo.CQI_UL_PerRB;
                   if ~isfield(obj.ChannelInfo,'TrafficUL'), traffic = zeros(obj.NumUEs,1); else, traffic = obj.ChannelInfo.TrafficUL; end
                   buffer = zeros(obj.NumUEs, 1); % Буфер UL на UE пока не моделируется
             end
             avgThr = obj.AvgThroughput; % Всегда доступно в объекте
             if ~isfield(obj.ChannelInfo,'RSRP'), rsrp = -140*ones(obj.NumUEs,1); else, rsrp = obj.ChannelInfo.RSRP; end

             % Пример признаков для RB: [CQIs всех UE на RB, AvgThrs всех UE, RSRPs всех UE, Buffers всех UE]
             numFeaturesPerRB = obj.NumUEs * 4;
             featuresPerRB = zeros(obj.NumRBs, numFeaturesPerRB);

             for rb = 1:obj.NumRBs
                 feat_cqi = cqiPerRB(:, rb).';        % [1 x NumUEs]
                 feat_avgThr = avgThr.';              % [1 x NumUEs]
                 feat_rsrp = rsrp.';                  % [1 x NumUEs]
                 feat_buffer = buffer.';              % [1 x NumUEs]
                 featuresPerRB(rb, :) = [feat_cqi, feat_avgThr, feat_rsrp, feat_buffer];
             end
        end

        function collectTrainingDataRB(obj, direction, allocatedGrid)
             featuresPerRB = obj.generateFeaturesPerRB(direction);
             if isempty(featuresPerRB), return; end
             targetUE_IDs = allocatedGrid;
             newData = [featuresPerRB, targetUE_IDs(:)];
             if isempty(obj.trainingDataRB)
                 obj.trainingDataRB = newData;
             else
                 obj.trainingDataRB = [obj.trainingDataRB; newData];
             end
             maxRows = 50000; % Ограничение
             if size(obj.trainingDataRB, 1) > maxRows
                 obj.trainingDataRB = obj.trainingDataRB(end-maxRows+1:end, :);
             end
         end

         function saveTrainingData(obj, direction)
              if isempty(obj.trainingDataRB)
                  fprintf('Нет данных для сохранения.\n'); return;
             end
             filename = sprintf('training_data_rb_%s.csv', direction);
             try
                writematrix(obj.trainingDataRB, filename);
                fprintf('Данные для обучения RB модели (%s) сохранены в %s (%d строк)\n', ...
                        direction, filename, size(obj.trainingDataRB, 1));
             catch ME
                 warning('Не удалось сохранить данные обучения (%s): %s', filename, ME.message);
             end
         end

        function applyMLDecisions(obj, decisionsMatrix, direction)
              if strcmpi(direction, 'DL')
                 targetGrid = zeros(1, obj.NumRBs);
                 for rb = 1:obj.NumRBs
                     ueForRB = find(decisionsMatrix(:, rb) == 1);
                     if ~isempty(ueForRB)
                         if length(ueForRB) > 1, ueForRB = ueForRB(randi(length(ueForRB))); end % Выбираем одного случайно
                         targetGrid(rb) = ueForRB;
                     end
                 end
                 obj.DL_Grid = targetGrid;
             else % UL
                  targetGrid = zeros(1, obj.NumRBs);
                  for rb = 1:obj.NumRBs
                      ueForRB = find(decisionsMatrix(:, rb) == 1);
                      if ~isempty(ueForRB)
                           if length(ueForRB) > 1, ueForRB = ueForRB(randi(length(ueForRB))); end
                          targetGrid(rb) = ueForRB;
                      end
                  end
                  obj.UL_Grid = targetGrid;
             end
         end

        function randomMatrix = generateRandomAllocationMatrix(obj)
              randomUE_per_RB = randi([0, obj.NumUEs], 1, obj.NumRBs);
              randomMatrix = zeros(obj.NumUEs, obj.NumRBs);
              for rb = 1:obj.NumRBs
                 ue_id = randomUE_per_RB(rb);
                 if ue_id > 0 && ue_id <= obj.NumUEs % Добавили проверку ue_id
                     randomMatrix(ue_id, rb) = 1;
                 end
              end
         end

    end % methods
end % classdef