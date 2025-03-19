classdef SystemLevelSimulator < handle
    properties
        NumUEs
        Bandwidth
        SCS
        NumRBs
        SimTime
        UELocations    % [x, y]
        RSRP           % Уровень сигнала
        SINR           % Качество сигнала
        CQI_DL         % Индикатор качества DL
        CQI_UL         % Индикатор качества UL
        gNBLocation    % Позиция БС
        ChannelInfo    % Информация о канале
        MovementStep = 8; % Шаг движения (м/с)
    end
    
    methods
        function obj = SystemLevelSimulator(numUEs, bw, scs, simTime)
            obj.NumUEs = numUEs;
            obj.Bandwidth = bw;
            obj.SCS = scs;
            obj.SimTime = simTime;
            obj.NumRBs = floor(bw/(12*scs));
            obj.gNBLocation = [250, 250];
            obj.initializeUE();
            obj.updateChannelConditions();
        end
        
        function initializeUE(obj)
            obj.UELocations = rand(obj.NumUEs, 2) * 500;
        end
        
        function updateNetwork(obj, ~)
            % Обновление позиций UE
            angles = rand(obj.NumUEs, 1) * 2*pi;
            deltaX = obj.MovementStep * cos(angles);
            deltaY = obj.MovementStep * sin(angles);
            obj.UELocations = mod(obj.UELocations + [deltaX, deltaY], 500);
            
            obj.updateChannelConditions();
        end
        
        function updateChannelConditions(obj)
            d = vecnorm(obj.UELocations - obj.gNBLocation, 2, 2);
            obj.RSRP = -112 - 35*log10(d + 1);
            noiseFloor = -174 + 10*log10(obj.Bandwidth);
            obj.SINR = obj.RSRP - noiseFloor;           
          
            % Расчет CQI
            obj.CQI_DL = discretize(obj.SINR, ...
                [-inf, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, inf], ...
                1:15);
            obj.CQI_UL = max(obj.CQI_DL - 1, 1);
            obj.CQI_UL = min(obj.CQI_UL, 15); % Добавлено ограничение сверху
            obj.CQI_UL(obj.SINR < -6) = 1;

            obj.ChannelInfo = struct(...
                'RSRP', obj.RSRP, ...
                'CQI_DL', obj.CQI_DL, ...
                'CQI_UL', obj.CQI_UL);
        end
    end
end