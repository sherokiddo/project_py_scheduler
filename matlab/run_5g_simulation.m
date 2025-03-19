clc; clear; close all;

%% Параметры симуляции
simTime = 20;              % Время симуляции (секунды)
numUEs = 5;                % Количество пользователей
bandwidth = 20e6;          % Ширина полосы (Гц)
scs = 30e3;                % Интервал между поднесущими (Гц)
numRBs = floor(bandwidth / (12 * scs));  
tddConfig = [1,1,1,1,1,1,1,0,0,0]; % TDD: 7DL 3UL

%% Инициализация компонентов
simulator = SystemLevelSimulator(numUEs, bandwidth, scs, simTime);
scheduler = CustomScheduler(...
    simulator.NumRBs, ...
    numUEs, ...
    'ProportionalFair', ...
    simulator.CQI_DL, ... % Передача CQI_DL
    simulator.CQI_UL);    % Передача CQI_UL

%% Настройка визуализации
figure('Position', [100 100 1200 800]);

% График перемещения UE
hAx1 = subplot(2,2,[1 3]);
hUEMarkers = scatter(hAx1, simulator.UELocations(:,1), simulator.UELocations(:,2), ...
    100, simulator.RSRP, 'filled', 'MarkerEdgeColor', 'k');
hold(hAx1, 'on');
hGNB = plot(hAx1, simulator.gNBLocation(1), simulator.gNBLocation(2), ...
    'rp', 'MarkerSize', 20, 'MarkerFaceColor', 'r');

% Линии соединения
gNBConnections = gobjects(numUEs, 1);
for ue = 1:numUEs
    gNBConnections(ue) = plot(hAx1, ...
        [simulator.UELocations(ue,1), simulator.gNBLocation(1)], ...
        [simulator.UELocations(ue,2), simulator.gNBLocation(2)], 'k--');
end
hold(hAx1, 'off');
title(hAx1, 'Передвижение UE');
xlabel(hAx1, 'X (м)'); ylabel(hAx1, 'Y (м)'); 
grid(hAx1, 'on'); colorbar(hAx1); caxis(hAx1, [-120 -80]);

% Ресурсная сетка DL
hAx2 = subplot(2,2,2);
hDLGrid = imagesc(hAx2, zeros(numUEs, numRBs));
title(hAx2, 'Ресурсная сетка DL');
xlabel(hAx2, 'Ресурсные блоки'); ylabel(hAx2, 'UE');
colormap(hAx2, [0.9 0.9 0.9; 0 0 1]); % Серый: свободно, Синий: DL
colorbar(hAx2, 'Ticks', [0,1], 'TickLabels', {'Свободно', 'DL'});

% Ресурсная сетка UL
hAx3 = subplot(2,2,4);
hULGrid = imagesc(hAx3, zeros(numUEs, numRBs));
title(hAx3, 'Ресурсная сетка UL');
xlabel(hAx3, 'Ресурсные блоки'); ylabel(hAx3, 'UE');
colormap(hAx3, [0.9 0.9 0.9; 1 0 0]); % Серый: свободно, Красный: UL
colorbar(hAx3, 'Ticks', [0,1], 'TickLabels', {'Свободно', 'UL'});

%% Главный цикл симуляции
for t = 1:simTime
    slotType = tddConfig(mod(t-1, length(tddConfig)) + 1);
    
    % Обновление модели
    simulator.updateNetwork(t);
    
    % Планирование ресурсов
    if slotType == 1
        scheduler.scheduleDL(simulator.ChannelInfo, t);
        gridData = scheduler.getDLResourceGrid();
        set(hDLGrid, 'CData', gridData);
    else
        scheduler.scheduleUL(simulator.ChannelInfo, t);
        gridData = scheduler.getULResourceGrid();
        set(hULGrid, 'CData', gridData);
    end

     if mod(t, 10) == 0 && strcmp(scheduler.SchedulerType, 'CatBoost')
        scheduler.saveTrainingData();
    end
    
    % Обновление визуализации UE
    set(hUEMarkers, 'XData', simulator.UELocations(:,1), ...
                    'YData', simulator.UELocations(:,2), ...
                    'CData', simulator.RSRP);
    
    % Обновление линий соединения
    for ue = 1:numUEs
        set(gNBConnections(ue), ...
            'XData', [simulator.UELocations(ue,1), simulator.gNBLocation(1)], ...
            'YData', [simulator.UELocations(ue,2), simulator.gNBLocation(2)]);
    end
    
    title(hAx1, sprintf('Позиции UE (t=%d сек)', t));
    pause(0.3);
    drawnow;
end