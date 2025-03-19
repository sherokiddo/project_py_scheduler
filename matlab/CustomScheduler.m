classdef CustomScheduler < handle
    properties
        % Конфигурация
        NumRBs                  % Количество ресурсных блоков
        NumUEs                  % Количество пользователей
        SchedulerType           % Тип планировщика
        SymbolsPerSlot          % OFDM символов в слоте
        
        % Ресурсные сетки
        DL_Grid                 % Ресурсная сетка DL [Symbols x RB x UE]
        UL_Grid                 % Ресурсная сетка UL [Symbols x RB x UE]
        
        % Состояние планировщика
        AvgThroughput           % Средняя пропускная способность
        CQI_DL                  % Индикатор качества канала DL
        CQI_UL                  % Индикатор качества канала UL
        HARQ_Processes          % HARQ процессы
        enableLogging = true;

        % ML-свойства
        mlEnabled = false;              % Включить ML-планировщик
        trainingData = table();          % Данные для обучения
        predictionCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    end
    
    methods
        function obj = CustomScheduler(numRBs, numUEs, type, cqi_dl, cqi_ul)
            % Конструктор класса
            obj.NumRBs = numRBs;
            obj.NumUEs = numUEs;
            obj.SchedulerType = type;
            obj.SymbolsPerSlot = 14;
            obj.CQI_DL = cqi_dl;
            obj.CQI_UL = max(cqi_ul, 1);
            
            % Инициализация ресурсных сеток
            obj.DL_Grid = zeros(obj.SymbolsPerSlot, numRBs, numUEs);
            obj.UL_Grid = zeros(obj.SymbolsPerSlot, numRBs, numUEs);
            
            % Инициализация состояния
            obj.AvgThroughput = ones(numUEs, 1);
            obj.HARQ_Processes = struct(...
                'UE', cell(8,1), ...
                'RB', cell(8,1), ...
                'Status', 'ACK');
        end
         
        function grid2D = getDLResourceGrid(obj)
            % Преобразует 3D-сетку DL в 2D (UE x RB)
            grid2D = squeeze(any(obj.DL_Grid ~= 0, 1))'; % Суммируем по символам и транспонируем
        end
        
        function grid2D = getULResourceGrid(obj)
            % Преобразует 3D-сетку UL в 2D (UE x RB)
            grid2D = squeeze(any(obj.UL_Grid ~= 0, 1))'; % Суммируем по символам и транспонируем
        end
        
        function scheduleDL(obj, channelInfo, t)
            % Планирование DL ресурсов
            obj.DL_Grid = zeros(size(obj.DL_Grid));
            if strcmp(obj.SchedulerType, 'CatBoost')
                obj.mlBasedScheduler(channelInfo);
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
            % Планирование UL ресурсов
            obj.UL_Grid = zeros(size(obj.UL_Grid));
            
            switch obj.SchedulerType
                case 'RoundRobin'
                    obj.roundRobinUL(t);
                case 'ProportionalFair'
                    obj.proportionalFairUL(channelInfo);
                case 'MaxThroughput'
                    obj.maxThroughputUL(channelInfo);
            end
            if obj.enableLogging
                obj.logSchedule('UL', t);
            end
        end
        
     function logSchedule(obj, direction, t)
            % Определение текущей сетки
            if strcmp(direction, 'DL')
                grid = obj.DL_Grid;
                cqi = obj.CQI_DL;
            else
                grid = obj.UL_Grid;
                cqi = obj.CQI_UL;
            end
            % Анализ выделенных ресурсов
            allocated = squeeze(any(grid ~= 0, 1))';  % UE x RB


            % Форматирование вывода
            fprintf('\n=== %s слот %d [Алгоритм: %s] ===\n', ...
                direction, t, obj.SchedulerType);
            
            for ue = 1:obj.NumUEs
                rbList = find(allocated(ue, :));
                if isempty(rbList)
                    fprintf('UE %d: Ресурсы не выделены (CQI=%d)\n', ue, cqi(ue));
                else
                    mcs = obj.getMCS(cqi(ue));
                    fprintf('UE %d: RB %s | MCS=%d (CQI=%d)\n', ...
                        ue, mat2str(rbList), mcs.code, cqi(ue));
                end
            end
        end

        function rbAlloc = allocateRB(obj, availableRBs)
            maxRB = min(12, numel(availableRBs)); % Гибкий размер блока
            if numel(availableRBs) < maxRB
                rbAlloc = [];
                return;
            end
            maxStart = numel(availableRBs) - maxRB + 1; % Вычисляем maxStart
            if maxStart < 1
                rbAlloc = [];
                return;
            end
            startIdx = randi([1, maxStart]);
            rbAlloc = availableRBs(startIdx:startIdx+ maxRB - 1);
        end
        
        
        function collectTrainingData(obj, channelInfo, decision)
            features = [channelInfo.CQI_DL, channelInfo.CQI_UL, channelInfo.RSRP];
            newRow = array2table([features, decision], 'VariableNames', {'CQI_DL', 'CQI_UL', 'RSRP', 'Decision'});
            obj.trainingData = [obj.trainingData; newRow];
        end
        
        function saveTrainingData(obj)
            writetable(obj.trainingData, 'training_data.csv');
        end
        
        function decisions = predictWithCatboost(obj, channelInfo)
            cacheKey = matlab.lang.MakeValidName(strcat(num2str(channelInfo.CQI_DL), num2str(channelInfo.CQI_UL), num2str(channelInfo.RSRP)));
            if obj.predictionCache.isKey(cacheKey)
                decisions = obj.predictionCache(cacheKey);
            else
                writematrix([channelInfo.CQI_DL, channelInfo.CQI_UL, channelInfo.RSRP], 'current_state.csv');
                system('python predict.py');
                decisions = readmatrix('predictions.csv');
                obj.predictionCache(cacheKey) = decisions;
            end
        end
        
        function mlBasedScheduler(obj, channelInfo)
            if isempty(obj.trainingData)
                decisions = randi([0, 1], obj.NumUEs, obj.NumRBs);
            else
                decisions = obj.predictWithCatboost(channelInfo);
            end
            for ue = 1:obj.NumUEs
                rbAlloc = find(decisions(ue, :));
                if ~isempty(rbAlloc)
                    obj.DL_Grid(4:end, rbAlloc, ue) = obj.getMCS(obj.CQI_DL(ue)).code;
                    obj.collectTrainingData(channelInfo, decisions(ue, :));
                end
            end
        end
        
        function processHARQFeedback(obj)
            % Обработка HARQ обратной связи
            for i = 1:numel(obj.HARQ_Processes)
                if strcmp(obj.HARQ_Processes(i).Status, 'NACK')
                    obj.retransmitProcess(i);
                end
            end
        end
        
        function retransmitProcess(obj, processID)
            % Повторная передача HARQ
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
            % Таблица MCS (TS 38.214)
            mcsTable = [
                1  0.1523; 2  0.2344; 3  0.3770; 4  0.6016; 5  0.8770;
                6  1.1758; 7  1.4766; 8  1.9141; 9  2.4063; 10 2.7305;
                11 3.3223; 12 3.9023; 13 4.5234; 14 5.1152; 15 5.5547;
            ];
            cqi = max(1, min(15, cqi));
            mcs.code = mcsTable(cqi, 1);
            mcs.efficiency = mcsTable(cqi, 2);
        end
        
        function roundRobinDL(obj, t)
            % Round Robin для DL
            startUE = mod(t-1, obj.NumUEs);
            rbsPerUE = floor(obj.NumRBs/obj.NumUEs);
            
            for ue = 1:obj.NumUEs
                idx = mod(startUE + ue-1, obj.NumUEs);
                rbStart = idx*rbsPerUE + 1;
                rbEnd = min((idx+1)*rbsPerUE, obj.NumRBs);
                
                mcs = obj.getMCS(obj.CQI_DL(ue));
                obj.DL_Grid(4:end, rbStart:rbEnd, ue) = mcs.code;
            end
        end
        
        function proportionalFairDL(obj, channelInfo)
            % Proportional Fair для DL
            metrics = (channelInfo.RSRP + 120)./(obj.AvgThroughput + 1e-6);
            [~, order] = sort(metrics, 'descend');
            availableRBs = 1:obj.NumRBs;
            
            for i = 1:obj.NumUEs
                ue = order(i);
                if ue < 1 || ue > obj.NumUEs, ...
                        continue; 
                end
                mcs = obj.getMCS(obj.CQI_DL(ue));
                rbAlloc = obj.allocateRB(availableRBs);
                
                if ~isempty(rbAlloc)
                    obj.DL_Grid(4:end, rbAlloc, ue) = mcs.code;
                    obj.AvgThroughput(ue) = 0.9*obj.AvgThroughput(ue) + ...
                        0.1*numel(rbAlloc)*mcs.efficiency;
                    availableRBs = setdiff(availableRBs, rbAlloc);
                end
            end
        end
        function maxThroughputDL(obj, channelInfo)
            % Max Throughput для DL (выделение ресурсов пользователям с лучшим CQI)
            [~, order] = sort(channelInfo.CQI_DL, 'descend');
            availableRBs = 1:obj.NumRBs;
            
            for i = 1:obj.NumUEs
                ue = order(i);
                if ue < 1 || ue > obj.NumUEs
                    continue;
                end
                mcs = obj.getMCS(channelInfo.CQI_DL(ue));
                rbAlloc = obj.allocateRB(availableRBs);
                
                if ~isempty(rbAlloc)
                    obj.DL_Grid(4:end, rbAlloc, ue) = mcs.code;
                    availableRBs = setdiff(availableRBs, rbAlloc);
                end
            end
        end
        
        function roundRobinUL(obj, t)
            % Round Robin для UL
            startUE = mod(t-1, obj.NumUEs);
            rbsPerUE = floor(obj.NumRBs/obj.NumUEs);
            
            for ue = 1:obj.NumUEs
                idx = mod(startUE + ue-1, obj.NumUEs);
                rbStart = idx*rbsPerUE + 1;
                rbEnd = min((idx+1)*rbsPerUE, obj.NumRBs);
                
                mcs = obj.getMCS(obj.CQI_UL(ue));
                obj.UL_Grid(4:end, rbStart:rbEnd, ue) = mcs.code;
            end
        end
        
        function proportionalFairUL(obj, channelInfo)
            % Proportional Fair для UL
            metrics = (channelInfo.RSRP + 120)./(obj.AvgThroughput + 1e-6);
            [~, order] = sort(metrics, 'descend');
            availableRBs = 1:obj.NumRBs;
            
            for i = 1:numel(order)
                ue = order(i);
                if ue > obj.NumUEs || ue < 1 % Проверка валидности индекса
                    continue; 
                end
                
                % Получение MCS с проверкой CQI
                cqi_ul = max(1, min(15, obj.CQI_UL(ue))); % Двойная проверка
                mcs = obj.getMCS(cqi_ul);
                rbAlloc = obj.allocateRB(availableRBs);
                
                if ~isempty(rbAlloc)
                    obj.UL_Grid(4:end, rbAlloc, ue) = mcs.code;
                    obj.AvgThroughput(ue) = 0.9*obj.AvgThroughput(ue) + ...
                        0.1*numel(rbAlloc)*mcs.efficiency;
                    availableRBs = setdiff(availableRBs, rbAlloc);
                    fprintf('UL: UE %d получил RB %d-%d\n', ue, rbAlloc(1), rbAlloc(end)); % Логирование
                end
            end
        end
         function maxThroughputUL(obj, channelInfo)
            % Max Throughput для UL
            [~, order] = sort(channelInfo.CQI_UL, 'descend');
            availableRBs = 1:obj.NumRBs;
            
            for i = 1:obj.NumUEs
                ue = order(i);
                if ue < 1 || ue > obj.NumUEs
                    continue;
                end
                mcs = obj.getMCS(channelInfo.CQI_UL(ue));
                rbAlloc = obj.allocateRB(availableRBs);
                
                if ~isempty(rbAlloc)
                    obj.UL_Grid(4:end, rbAlloc, ue) = mcs.code;
                    availableRBs = setdiff(availableRBs, rbAlloc);
                end
            end  
        end

        function dlGrid = getDLGrid(obj)
            % Получение DL сетки
            dlGrid = sum(obj.DL_Grid, 3);
        end
        
        function ulGrid = getULGrid(obj)
            % Получение UL сетки
            ulGrid = sum(obj.UL_Grid, 3);
        end
    end
end