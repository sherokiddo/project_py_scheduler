classdef CustomScheduler < handle
    properties
        % Конфигурация
        NumRBs         % Количество ресурсных блоков
        NumUEs         % Количество пользователей
        SchedulerType  % Тип планировщика (например, 'CatBoost')
        SymbolsPerSlot % Количество OFDM символов в слоте
        
        % Ресурсные сетки
        DL_Grid  % Матрица DL [Symbols x RB x UE]
        UL_Grid  % Матрица UL [Symbols x RB x UE]
        
        % Состояние планировщика
        AvgThroughput   % Средняя пропускная способность для UE
        CQI_DL          % CQI для DL
        CQI_UL          % CQI для UL
        HARQ_Processes  % Структура для HARQ
        enableLogging = true;
        
        % ML-свойства
        mlEnabled = true;
        trainingData = [];  % накопление обучающих данных
        predictionCache = containers.Map('KeyType','char','ValueType','any');
        
        % Для ускорения ML-вызовов
        lastMLTime = -inf;
        mlInterval = 3;  % обновлять не чаще чем раз в 3 слота
    end
    methods
        function obj = CustomScheduler(numRBs, numUEs, type, cqi_dl, cqi_ul)
            obj.NumRBs = numRBs;
            obj.NumUEs = numUEs;
            obj.SchedulerType = type;
            obj.SymbolsPerSlot = 14;
            obj.CQI_DL = cqi_dl;
            obj.CQI_UL = max(cqi_ul, 1);
            obj.DL_Grid = zeros(obj.SymbolsPerSlot, numRBs, numUEs);
            obj.UL_Grid = zeros(obj.SymbolsPerSlot, numRBs, numUEs);
            obj.AvgThroughput = ones(numUEs,1);
            obj.HARQ_Processes = struct('UE', cell(8,1), 'RB', cell(8,1), 'Status', 'ACK');
        end
        
        function grid2D = getDLResourceGrid(obj)
            grid2D = squeeze(any(obj.DL_Grid ~= 0, 1))';
        end
        
        function grid2D = getULResourceGrid(obj)
            grid2D = squeeze(any(obj.UL_Grid ~= 0, 1))';
        end
        
        % Новые методы: получение текущего распределения по RB для визуализации.
        function alloc = getCurrentDLAllocation(obj)
            % Вернуть строку [1 x NumRB]: если для RB i выделен ресурс UE k, alloc(i)=k, иначе 0.
            alloc = zeros(1, obj.NumRBs);
            % Суммируем по символам: получаем [RB x UE]
            grid = squeeze(sum(obj.DL_Grid, 1));
            for ue = 1:obj.NumUEs
                idx = find(grid(:, ue) ~= 0);
                alloc(idx) = ue;
            end
        end
        
        function alloc = getCurrentULAllocation(obj)
            alloc = zeros(1, obj.NumRBs);
            grid = squeeze(sum(obj.UL_Grid, 1));
            for ue = 1:obj.NumUEs
                idx = find(grid(:, ue) ~= 0);
                alloc(idx) = ue;
            end
        end
        
        function scheduleDL(obj, channelInfo, t)
            obj.DL_Grid = zeros(size(obj.DL_Grid));
            if strcmp(obj.SchedulerType, 'CatBoost')
                if (t - obj.lastMLTime) >= obj.mlInterval || isempty(obj.predictionCache)
                    obj.mlBasedScheduler(channelInfo);
                    obj.lastMLTime = t;
                else
                    decisions = obj.predictionCache('lastDL');
                    obj.applyDecisionsDL(decisions);
                end
            else
                switch obj.SchedulerType
                    case 'RoundRobin'
                        obj.roundRobinDL(t);
                    case 'ProportionalFair'
                        obj.proportionalFairDL(channelInfo);
                    case 'MaxThroughput'
                        obj.maxThroughputDL(channelInfo);
                end
            end
            if obj.enableLogging
                obj.logSchedule('DL', t);
            end
            obj.processHARQFeedback();
        end
        
        function scheduleUL(obj, channelInfo, t)
            obj.UL_Grid = zeros(size(obj.UL_Grid));
            if strcmp(obj.SchedulerType, 'CatBoost')
                if (t - obj.lastMLTime) >= obj.mlInterval || isempty(obj.predictionCache)
                    obj.mlBasedSchedulerUL(channelInfo);
                    obj.lastMLTime = t;
                else
                    decisions = obj.predictionCache('lastUL');
                    obj.applyDecisionsUL(decisions);
                end
            else
                switch obj.SchedulerType
                    case 'RoundRobin'
                        obj.roundRobinUL(t);
                    case 'ProportionalFair'
                        obj.proportionalFairUL(channelInfo);
                    case 'MaxThroughput'
                        obj.maxThroughputUL(channelInfo);
                end
            end
            if obj.enableLogging
                obj.logSchedule('UL', t);
            end
        end
        
        function logSchedule(obj, direction, t)
            if strcmp(direction, 'DL')
                grid = obj.DL_Grid;
                cqi = obj.CQI_DL;
            else
                grid = obj.UL_Grid;
                cqi = obj.CQI_UL;
            end
            allocated = squeeze(any(grid ~= 0, 1))';
            fprintf('\n=== %s слот %d [Алгоритм: %s] ===\n', direction, t, obj.SchedulerType);
            for ue = 1:obj.NumUEs
                rbList = find(allocated(ue,:));
                if isempty(rbList)
                    fprintf('UE %d: Ресурсы не выделены (CQI=%d)\n', ue, cqi(ue));
                else
                    mcs = obj.getMCS(cqi(ue));
                    fprintf('UE %d: RB %s | MCS=%d (CQI=%d)\n', ue, mat2str(rbList), mcs.code, cqi(ue));
                end
            end
        end
        
        function rbAlloc = allocateRB(obj, availableRBs)
            maxRB = min(12, numel(availableRBs));
            if numel(availableRBs) < maxRB
                rbAlloc = [];
                return;
            end
            maxStart = numel(availableRBs)-maxRB+1;
            if maxStart < 1
                rbAlloc = [];
                return;
            end
            startIdx = randi([1, maxStart]);
            rbAlloc = availableRBs(startIdx:startIdx+maxRB-1);
        end
        
        function collectTrainingData(obj, channelInfo, decision)
            features = [channelInfo.CQI_DL, channelInfo.CQI_UL, channelInfo.RSRP];
            features = [features, channelInfo.TrafficDL, channelInfo.TrafficUL];
            newRows = [features, decision];
            if isempty(obj.trainingData)
                obj.trainingData = newRows;
            else
                obj.trainingData = [obj.trainingData; newRows];
            end
        end
        
        function saveTrainingData(obj)
            if ~isempty(obj.trainingData)
                writematrix(obj.trainingData, 'training_data.csv');
                fprintf('Training data сохранены в training_data.csv\n');
            end
        end
        
        function decisions = predictWithCatboost(obj, channelInfo)
            cacheKey = matlab.lang.makeValidName(strcat(num2str(channelInfo.CQI_DL(1)), '_',...
                num2str(channelInfo.CQI_UL(1)), '_', num2str(channelInfo.RSRP(1))));
            if obj.predictionCache.isKey(cacheKey)
                decisions = obj.predictionCache(cacheKey);
                return;
            end
            features = [channelInfo.CQI_DL, channelInfo.CQI_UL, channelInfo.RSRP];
            features = [features, channelInfo.TrafficDL, channelInfo.TrafficUL];
            writematrix(features, 'current_state.csv');
            [status,~] = system(sprintf('python catboost_scheduler.py %d', obj.NumRBs));
            if status ~= 0
                disp('Ошибка вызова Python‑скрипта. Используем случайные решения.');
                decisions = randi([0,1],obj.NumUEs,obj.NumRBs);
            else
                decisions = csvread('predictions.csv');
            end
            obj.predictionCache(cacheKey) = decisions;
            obj.predictionCache('lastDL') = decisions;
            obj.predictionCache('lastUL') = decisions;
        end
        
        function applyDecisionsDL(obj, decisions)
            for ue = 1:obj.NumUEs
                rbAlloc = find(decisions(ue,:) == 1);
                if ~isempty(rbAlloc)
                    obj.DL_Grid(4:end, rbAlloc, ue) = obj.getMCS(obj.CQI_DL(ue)).code;
                end
            end
        end
        
        function applyDecisionsUL(obj, decisions)
            for ue = 1:obj.NumUEs
                rbAlloc = find(decisions(ue,:) == 1);
                if ~isempty(rbAlloc)
                    obj.UL_Grid(4:end, rbAlloc, ue) = obj.getMCS(obj.CQI_UL(ue)).code;
                end
            end
        end
        
        function mlBasedScheduler(obj, channelInfo)
            if isempty(obj.trainingData)
                decisions = randi([0,1],obj.NumUEs,obj.NumRBs);
            else
                decisions = obj.predictWithCatboost(channelInfo);
            end
            obj.applyDecisionsDL(decisions);
            obj.collectTrainingData(channelInfo, decisions);
        end
        
        function mlBasedSchedulerUL(obj, channelInfo)
            if isempty(obj.trainingData)
                decisions = randi([0,1],obj.NumUEs,obj.NumRBs);
            else
                decisions = obj.predictWithCatboost(channelInfo);
            end
            obj.applyDecisionsUL(decisions);
            obj.collectTrainingData(channelInfo, decisions);
        end
        
        function processHARQFeedback(obj)
            for i = 1:numel(obj.HARQ_Processes)
                if strcmp(obj.HARQ_Processes(i).Status, 'NACK')
                    obj.retransmitProcess(i);
                end
            end
        end
        
        function retransmitProcess(obj, processID)
            if processID < 1 || processID > numel(obj.HARQ_Processes)
                return;
            end
            ue = obj.HARQ_Processes(processID).UE;
            rbs = obj.HARQ_Processes(processID).RB;
            if ue < 1 || ue > obj.NumUEs || isempty(rbs)
                return;
            end
            obj.DL_Grid(4:end, rbs, ue) = obj.getMCS(obj.CQI_DL(ue)).code;
        end
        
        function mcs = getMCS(~, cqi)
            mcsTable = [...
                1  0.1523;
                2  0.2344;
                3  0.3770;
                4  0.6016;
                5  0.8770;
                6  1.1758;
                7  1.4766;
                8  1.9141;
                9  2.4063;
                10 2.7305;
                11 3.3223;
                12 3.9023;
                13 4.5234;
                14 5.1152;
                15 5.5547];
            cqi = max(1, min(15, cqi));
            mcs.code = mcsTable(cqi, 1);
            mcs.efficiency = mcsTable(cqi, 2);
        end
        
        function roundRobinDL(obj, t)
            startUE = mod(t-1, obj.NumUEs);
            rbsPerUE = floor(obj.NumRBs/obj.NumUEs);
            for ue = 1:obj.NumUEs
                idx = mod(startUE+ue-1, obj.NumUEs);
                rbStart = idx*rbsPerUE + 1;
                rbEnd = min((idx+1)*rbsPerUE, obj.NumRBs);
                mcs = obj.getMCS(obj.CQI_DL(ue));
                obj.DL_Grid(4:end, rbStart:rbEnd, ue) = mcs.code;
            end
        end
        
        function proportionalFairDL(obj, channelInfo)
            metrics = (channelInfo.RSRP+120)./(obj.AvgThroughput+1e-6);
            [~, order] = sort(metrics, 'descend');
            availableRBs = 1:obj.NumRBs;
            for i = 1:obj.NumUEs
                ue = order(i);
                if ue < 1 || ue > obj.NumUEs, continue; end
                mcs = obj.getMCS(obj.CQI_DL(ue));
                rbAlloc = obj.allocateRB(availableRBs);
                if ~isempty(rbAlloc)
                    obj.DL_Grid(4:end, rbAlloc, ue) = mcs.code;
                    obj.AvgThroughput(ue) = 0.9*obj.AvgThroughput(ue) + 0.1*numel(rbAlloc)*mcs.efficiency;
                    availableRBs = setdiff(availableRBs, rbAlloc);
                end
            end
        end
        
        function maxThroughputDL(obj, channelInfo)
            [~, order] = sort(channelInfo.CQI_DL, 'descend');
            availableRBs = 1:obj.NumRBs;
            for i = 1:obj.NumUEs
                ue = order(i);
                if ue < 1 || ue > obj.NumUEs, continue; end
                mcs = obj.getMCS(channelInfo.CQI_DL(ue));
                rbAlloc = obj.allocateRB(availableRBs);
                if ~isempty(rbAlloc)
                    obj.DL_Grid(4:end, rbAlloc, ue) = mcs.code;
                    availableRBs = setdiff(availableRBs, rbAlloc);
                end
            end
        end
        
        function roundRobinUL(obj, t)
            startUE = mod(t-1, obj.NumUEs);
            rbsPerUE = floor(obj.NumRBs/obj.NumUEs);
            for ue = 1:obj.NumUEs
                idx = mod(startUE+ue-1, obj.NumUEs);
                rbStart = idx*rbsPerUE + 1;
                rbEnd = min((idx+1)*rbsPerUE, obj.NumRBs);
                mcs = obj.getMCS(obj.CQI_UL(ue));
                obj.UL_Grid(4:end, rbStart:rbEnd, ue) = mcs.code;
            end
        end
        
        function proportionalFairUL(obj, channelInfo)
            metrics = (channelInfo.RSRP+120)./(obj.AvgThroughput+1e-6);
            [~, order] = sort(metrics, 'descend');
            availableRBs = 1:obj.NumRBs;
            for i = 1:numel(order)
                ue = order(i);
                if ue > obj.NumUEs || ue < 1, continue; end
                cqi_ul = max(1, min(15, obj.CQI_UL(ue)));
                mcs = obj.getMCS(cqi_ul);
                rbAlloc = obj.allocateRB(availableRBs);
                if ~isempty(rbAlloc)
                    obj.UL_Grid(4:end, rbAlloc, ue) = mcs.code;
                    obj.AvgThroughput(ue) = 0.9*obj.AvgThroughput(ue) + 0.1*numel(rbAlloc)*mcs.efficiency;
                    availableRBs = setdiff(availableRBs, rbAlloc);
                    fprintf('UL: UE %d получил RB %d-%d\n', ue, rbAlloc(1), rbAlloc(end));
                end
            end
        end
        
        function maxThroughputUL(obj, channelInfo)
            [~, order] = sort(channelInfo.CQI_UL, 'descend');
            availableRBs = 1:obj.NumRBs;
            for i = 1:obj.NumUEs
                ue = order(i);
                if ue < 1 || ue > obj.NumUEs, continue; end
                mcs = obj.getMCS(channelInfo.CQI_UL(ue));
                rbAlloc = obj.allocateRB(availableRBs);
                if ~isempty(rbAlloc)
                    obj.UL_Grid(4:end, rbAlloc, ue) = mcs.code;
                    availableRBs = setdiff(availableRBs, rbAlloc);
                end
            end
        end
        
        function dlGrid = getDLGrid(obj)
            dlGrid = sum(obj.DL_Grid, 3);
        end
        
        function ulGrid = getULGrid(obj)
            ulGrid = sum(obj.UL_Grid, 3);
        end
    end
end