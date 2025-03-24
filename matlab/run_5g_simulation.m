% scheduler_simulation.m
clc; clear; close all;

%% Параметры симуляции
simTime = 20;              % Время симуляции (сек)
numUEs = 5;                % Количество пользователей (UE)
bandwidth = 20e6;          % Ширина полосы (Гц)
scs = 30e3;                % Поднесущая (Гц)
numRBs = floor(bandwidth/(12*scs));  % Количество ресурсных блоков
tddConfig = [1,1,1,1,1,1,1,0,0,0];    % TDD: 7 слотов DL, 3 слота UL

%% Создание объектов симулятора и планировщика
simulator = SystemLevelSimulator(numUEs, bandwidth, scs, simTime);
scheduler  = CustomScheduler(simulator.NumRBs, numUEs, 'MaxThroughput', simulator.CQI_DL, simulator.CQI_UL);

%% Настройка визуализации
% Окно для отображения мобильности UE
hFigMobility = figure('Name', 'Мобильность UE','Position',[100 100 1200 800]);
hAxMobility = axes('Parent', hFigMobility);
hUEMarkers = scatter(hAxMobility, simulator.UELocations(:,1), simulator.UELocations(:,2), 100, simulator.RSRP, 'filled','MarkerEdgeColor','k');
hold(hAxMobility, 'on');
hGNB = plot(hAxMobility, simulator.gNBLocation(1), simulator.gNBLocation(2), 'rp', 'MarkerSize',20, 'MarkerFaceColor','r');
gNBConnections = gobjects(numUEs,1);
for ue = 1:numUEs
    gNBConnections(ue) = plot(hAxMobility, [simulator.UELocations(ue,1), simulator.gNBLocation(1)], ...
        [simulator.UELocations(ue,2), simulator.gNBLocation(2)], 'k--');
end
hold(hAxMobility, 'off');
title(hAxMobility, 'Передвижение UE');
xlabel(hAxMobility, 'X (м)'); ylabel(hAxMobility, 'Y (м)');
axis(hAxMobility, [0 500 0 500]);
grid(hAxMobility, 'on'); colorbar(hAxMobility); caxis(hAxMobility, [-120 -80]);

% Окно для ресурсной сетки, где строки – временные слоты, столбцы – RB
hFigGrid = figure('Name','Ресурсная сетка','Position',[1500 100 600 800]);
resourceGridTime = zeros(simTime, numRBs);
hResGrid = imagesc(resourceGridTime);
title('Ресурсная сетка (слот vs RB)');
xlabel('Ресурсные блоки');
ylabel('Слот');
% Задаем собственную палитру:
% 0 - свободно (белый);
% 1...numUEs - DL (цвета из lines);
% (offset+1)...(offset+numUEs) - UL (отличаются смещением)
offset = 20;
freeColor = [1 1 1];
dlColors = lines(numUEs);
ulColors = lines(numUEs)*0.8; % немного темнее для UL
customCmap = [freeColor; dlColors; ulColors];
colormap(gca, customCmap);
caxis(gca, [0 offset+numUEs]); 
colorbar(gca, 'Ticks', [0, 1:(numUEs), (offset+1):(offset+numUEs)], ...
    'TickLabels', [{'Свободно'}, arrayfun(@(x) sprintf('DL-UE%d', x), 1:numUEs, 'UniformOutput', false), ...
    arrayfun(@(x) sprintf('UL-UE%d', x), 1:numUEs, 'UniformOutput', false)]);

%% Главный цикл симуляции
if isempty(gcp('nocreate'))
    parpool;
end

for t = 1:simTime
    % Обновляем состояние системы: мобильность, трафик, канал
    simulator.updateNetwork(t);
    
    % Выбор слота по TDD: если DL (==1) – используем DL-алгоритм и получаем распределение,
    % иначе UL – сдвигаем значения на offset.
    if tddConfig(mod(t-1, numel(tddConfig)) + 1) == 1  % DL слот
        scheduler.scheduleDL(simulator.ChannelInfo, t);
        allocRow = scheduler.getCurrentDLAllocation();  % значения от 1 до numUEs
    else  % UL слот
        scheduler.scheduleUL(simulator.ChannelInfo, t);
        allocRow = scheduler.getCurrentULAllocation();  % значения от 1 до numUEs
        allocRow = allocRow + offset;  % смещаем для UL
    end
    resourceGridTime(t, :) = allocRow;
    
    % Обновляем отображение мобильности UE
    set(hUEMarkers, 'XData', simulator.UELocations(:,1), 'YData', simulator.UELocations(:,2), 'CData', simulator.RSRP);
    for ue = 1:numUEs
        set(gNBConnections(ue), 'XData', [simulator.UELocations(ue,1), simulator.gNBLocation(1)], ...
            'YData', [simulator.UELocations(ue,2), simulator.gNBLocation(2)]);
    end
    title(hAxMobility, sprintf('Передвижение UE (t = %d сек)', t));
    
    % Обновляем окно ресурсной сетки; если окно было закрыто – создаём заново.
    if ~ishandle(hResGrid)
        warning('Окно ресурсной сетки было закрыто. Воссоздаем его.');
        hFigGrid = figure('Name','Ресурсная сетка','Position',[1500 100 600 800]);
        hResGrid = imagesc(resourceGridTime);
        title('Ресурсная сетка (слот vs RB)');
        xlabel('Ресурсные блоки'); ylabel('Слот');
        colormap(gca, customCmap);
        caxis(gca, [0 offset+numUEs]);
        colorbar(gca, 'Ticks', [0, 1:(numUEs), (offset+1):(offset+numUEs)], ...
            'TickLabels', [{'Свободно'}, arrayfun(@(x) sprintf('DL-UE%d', x), 1:numUEs, 'UniformOutput', false), ...
            arrayfun(@(x) sprintf('UL-UE%d', x), 1:numUEs, 'UniformOutput', false)]);
    else
        set(hResGrid, 'CData', resourceGridTime);
    end
    
    drawnow limitrate;
    pause(0.2);
end