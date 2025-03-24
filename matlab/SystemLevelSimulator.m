classdef SystemLevelSimulator < handle
    properties
        NumUEs          % Количество UE
        Bandwidth       % Ширина полосы (Гц)
        SCS             % Поднесущая (Гц)
        NumRBs          % Количество ресурсных блоков
        SimTime         % Время симуляции (сек)
        UELocations     % Позиции UE [x, y]
        RSRP            % Уровень сигнала (дБм)
        SINR            % Соотношение сигнал/шум (дБ)
        CQI_DL          % CQI для DL
        CQI_UL          % CQI для UL
        gNBLocation     % Позиция базовой станции [x, y]
        ChannelInfo     % Структура с информацией о канале
        MovementStep = 8;   % Шаг движения (м)
        
        % Трафик-модель (случайная нагрузка)
        TrafficDL       % Трафик для DL (например, число пакетов в секунду)
        TrafficUL       % Трафик для UL
        lambdaDL = 5;    % Среднее число пакетов в секунду для DL
        lambdaUL = 3;    % Среднее число пакетов в секунду для UL
        
        % Модель мобильности – Random Waypoint
        Destination     % Текущие целевые точки для UE [NumUEs x 2]
    end
    
    methods
        function obj = SystemLevelSimulator(numUEs, bw, scs, simTime)
            obj.NumUEs = numUEs;
            obj.Bandwidth = bw;
            obj.SCS = scs;
            obj.SimTime = simTime;
            obj.NumRBs = floor(bw / (12*scs));
            obj.gNBLocation = [250, 250];
            obj.initializeUE();
            obj.initializeMobility();
            obj.updateChannelConditions();
            obj.updateTraffic();  % задаём начальные значения трафика
        end
        
        function initializeUE(obj)
            % Начальные позиции UE равномерно по территории 500x500
            obj.UELocations = rand(obj.NumUEs,2) * 500;
        end
        
        function initializeMobility(obj)
            % Для каждой UE выбираем случайную целевую точку (Random Waypoint)
            obj.Destination = rand(obj.NumUEs, 2) * 500;
        end
        
        function updateTraffic(obj)
            % Генерация трафика для DL и UL по пуассоновскому процессу
            obj.TrafficDL = poissrnd(obj.lambdaDL, obj.NumUEs, 1);
            obj.TrafficUL = poissrnd(obj.lambdaUL, obj.NumUEs, 1);
        end
        
        function updateNetwork(obj, ~)
            % Обновляем мобильность по модели Random Waypoint:
            for ue = 1:obj.NumUEs
                currentPos = obj.UELocations(ue,:);
                dest = obj.Destination(ue,:);
                direction = dest - currentPos;
                dist = norm(direction);
                if dist < obj.MovementStep
                    % Если цель достигнута – выбираем новую случайную цель
                    obj.Destination(ue,:) = rand(1,2)*500;
                else
                    % Двигаемся на шаг MovementStep по направлению к цели
                    obj.UELocations(ue,:) = currentPos + (direction/dist)*obj.MovementStep;
                end
            end
            % Обновляем трафик и канальные условия после перемещения
            obj.updateTraffic();
            obj.updateChannelConditions();
        end
        
        function updateChannelConditions(obj)
            % Вычисляем расстояния от UE до gNB:
            d = vecnorm(obj.UELocations - obj.gNBLocation, 2, 2);
            
            % Основное затухание (path loss): 
            pathLoss = -112 - 35*log10(d + 1);
            
            % Теневое затухание – лог-нормальное с σ = 8 дБ
            shadowFading = 8 * randn(size(d));
            
            % Быстрое (Rayleigh) затухание
            fastFading = 20 * log10(abs((1/sqrt(2)) * (randn(size(d)) + 1j*randn(size(d)))));
            
            % Итоговый уровень сигнала RSRP (в дБм)
            obj.RSRP = pathLoss + shadowFading + fastFading;
            
            % Расчет уровня шума
            noiseFloor = -174 + 10 * log10(obj.Bandwidth);
            obj.SINR = obj.RSRP - noiseFloor;
            
            % Определяем CQI для DL (значения от 1 до 15)
            obj.CQI_DL = discretize(obj.SINR, [-inf, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, inf], 1:15);
            obj.CQI_UL = max(obj.CQI_DL - 1, 1);
            obj.CQI_UL = min(obj.CQI_UL, 15);
            obj.CQI_UL(obj.SINR < -6) = 1;
            
            % Собираем всю информацию в структуру ChannelInfo, добавляя трафик
            obj.ChannelInfo = struct('RSRP', obj.RSRP, ...
                'CQI_DL', obj.CQI_DL, ...
                'CQI_UL', obj.CQI_UL, ...
                'TrafficDL', obj.TrafficDL, ...
                'TrafficUL', obj.TrafficUL);
        end
    end
end