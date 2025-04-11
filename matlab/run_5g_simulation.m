clc; clear; close all;

%% --- ОСНОВНЫЕ НАСТРОЙКИ СИМУЛЯЦИИ ---
simTime = 20;              % Время симуляции (сек)
numUEs = 5;                % Количество пользователей (UE)
bandwidth = 20e6;          % Ширина полосы (Гц)
scs = 30e3;                % Поднесущая (Гц)
carrierFrequency = 3.5e9;  % Несущая частота (Гц) - ВАЖНО для 3GPP каналов
gNBTxPower = 46;           % Пример: Мощность передачи gNB (дБм) - для будущего расчета DL SINR
ueTxPower = 23;            % Пример: Мощность передачи UE (дБм)

% --- Выбор Моделей ---
mobilityModel = 'GaussMarkov'; % 'RandomWaypoint' или 'GaussMarkov'
channelModel = 'Custom'; % 'Custom', 'TDL_UMa_NLOS', 'TDL_RMa_NLOS' (или другие TDL профили)

% --- Параметры TDD ---
tddConfig = [1,1,1,1,1,1,1,0,0,0]; % TDD: 7 слотов DL, 3 слота UL (пример)
slotsPerPattern = length(tddConfig);

%% Расчет зависимых параметров
numRBs = floor(bandwidth / (12 * scs));

%% Создание объектов симулятора и планировщика
fprintf('Инициализация симулятора...\n');
simulator = SystemLevelSimulator(numUEs, bandwidth, scs, simTime, carrierFrequency, mobilityModel, channelModel, gNBTxPower, ueTxPower);
fprintf('Инициализация планировщика...\n');
% Используем планировщик, который может работать с CQI per RB (например, обновленный MaxThroughput)
scheduler = CustomScheduler(simulator.NumRBs, numUEs, 'CatBoost'); % Начальные CQI не так важны

%% Настройка визуализации мобильности UE
hFigMobility = figure('Name', 'Мобильность UE','Position',[100 100 800 600]);
hAxMobility = axes('Parent', hFigMobility);
hUEMarkers = scatter(hAxMobility, simulator.UELocations(:,1), simulator.UELocations(:,2), 100, 'b', 'filled','MarkerEdgeColor','k'); % RSRP может быть недоступен сразу
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
axis(hAxMobility, [0 simulator.MapSize(1) 0 simulator.MapSize(2)]);
grid(hAxMobility, 'on'); colorbar(hAxMobility); % caxis нужно будет настроить под RSRP, когда он появится

%% Настройка визуализации ресурсных сеток DL и UL
resourceGridDL = zeros(numRBs, simTime);
resourceGridUL = zeros(numRBs, simTime);
offset = numUEs + 5; % Сдвиг для UL, чтобы цвета не пересекались с DL UE ID
freeColor = [1 1 1]; % Белый
dlColors = lines(numUEs);
% Генерируем цвета UL так, чтобы они отличались от DL, если numUEs большой
if numUEs <= 7
    ulColors = cool(numUEs);
else
    ulColors = parula(numUEs); % Другая палитра
end
customCmap = [freeColor; dlColors; ulColors];

% Окно для ресурсной сетки DL
hFigGridDL = figure('Name','Ресурсная сетка DL','Position',[950 550 600 400]);
hAxGridDL = axes('Parent', hFigGridDL);
hResGridDL = imagesc(hAxGridDL, resourceGridDL);
xlabel(hAxGridDL,'Слот (TTI)'); ylabel(hAxGridDL,'Ресурсный блок');
title(hAxGridDL,'Ресурсная сетка DL');
colormap(hAxGridDL, customCmap); caxis(hAxGridDL,[0 offset + numUEs]); colorbar(hAxGridDL);

% Окно для ресурсной сетки UL
hFigGridUL = figure('Name','Ресурсная сетка UL','Position',[950 100 600 400]);
hAxGridUL = axes('Parent', hFigGridUL);
hResGridUL = imagesc(hAxGridUL, resourceGridUL);
xlabel(hAxGridUL,'Слот (TTI)'); ylabel(hAxGridUL,'Ресурсный блок');
title(hAxGridUL,'Ресурсная сетка UL');
colormap(hAxGridUL, customCmap); caxis(hAxGridUL,[0 offset + numUEs]); colorbar(hAxGridUL);

%% Инициализация истории для графиков (Опционально)
sinrHistoryUL = zeros(numUEs, simTime);
% sinrHistoryDL = zeros(numUEs, simTime); % Для будущего DL SINR
pathLossHistory = zeros(numUEs, simTime);
timeVector = 1:simTime;

%% Главный цикл симуляции
fprintf('Запуск симуляции...\n');
for t = 1:simTime
    fprintf('Время t = %d / %d\n', t, simTime);

    % Обновляем состояние системы: мобильность, трафик, канал
    % Передаем планировщик для доступа к previous allocations
    simulator.updateNetwork(t, scheduler);

    % --- Сохранение данных для истории (Опционально) ---
    % Используем средний SINR для простоты графика
    if isfield(simulator.ChannelInfo, 'SINR_UL_Avg')
        sinrHistoryUL(:, t) = simulator.ChannelInfo.SINR_UL_Avg;
    end
     if isfield(simulator.ChannelInfo, 'PathLoss_dB')
        pathLossHistory(:, t) = simulator.ChannelInfo.PathLoss_dB;
     end
    % --- Конец сохранения данных ---

    % --- Обновление планировщика свежими данными канала ---
    % Передаем всю структуру ChannelInfo, планировщик сам возьмет нужное
    scheduler.updateChannelInfo(simulator.ChannelInfo);

    % --- Выбор слота по TDD и планирование ---
    slotType = tddConfig(mod(t-1, slotsPerPattern) + 1);

    if slotType == 1  % DL слот
        scheduler.scheduleDL(t); % Передаем только время
        allocRow = scheduler.getCurrentDLAllocation(); % Возвращает ID UE (1..numUEs) или 0
        fprintf('DL Слот %d Распределение: %s\n', t, mat2str(allocRow)); % <-- ОТЛАДКА
        resourceGridDL(:, t) = allocRow(:);
    else  % UL слот
        scheduler.scheduleUL(t); % Передаем только время
        allocRow = scheduler.getCurrentULAllocation(); % Возвращает ID UE (1..numUEs) или 0
        fprintf('UL Слот %d Распределение (до offset): %s\n', t, mat2str(allocRow)); % <-- ОТЛАДКА
        % Применяем offset для визуализации UL
        nonZeroIdx = allocRow > 0;
        allocRow(nonZeroIdx) = allocRow(nonZeroIdx) + offset;
        resourceGridUL(:, t) = allocRow(:);
    end

    % --- Обновление визуализации ---
    % Мобильность
    if ishandle(hFigMobility) && ishandle(hUEMarkers) && isvalid(hUEMarkers)
        set(hUEMarkers, 'XData', simulator.UELocations(:,1), 'YData', simulator.UELocations(:,2));
        % Обновляем цвет на основе RSRP, если он есть
        if isfield(simulator.ChannelInfo, 'RSRP') && ~isempty(simulator.ChannelInfo.RSRP)
             set(hUEMarkers, 'CData', simulator.ChannelInfo.RSRP);
             caxis(hAxMobility, [min(simulator.ChannelInfo.RSRP)-5, max(simulator.ChannelInfo.RSRP)+5]); % Автонастройка colorbar
        end
        for ue = 1:numUEs
            if ishandle(gNBConnections(ue)) && isvalid(gNBConnections(ue))
                 set(gNBConnections(ue), 'XData', [simulator.UELocations(ue,1), simulator.gNBLocation(1)], ...
                     'YData', [simulator.UELocations(ue,2), simulator.gNBLocation(2)]);
            end
        end
       title(hAxMobility, sprintf('Передвижение UE (t = %d сек)', t));
    else
        warning('Окно мобильности было закрыто.');
        break; % Прервать цикл
    end

    % Ресурсные сетки
    if ishandle(hResGridDL) && isvalid(hResGridDL)
        set(hResGridDL, 'CData', resourceGridDL);
         updateAnnotations(hAxGridDL, resourceGridDL, 0); % Можно добавить, если не тормозит
    end
    if ishandle(hResGridUL) && isvalid(hResGridUL)
        set(hResGridUL, 'CData', resourceGridUL);
         updateAnnotations(hAxGridUL, resourceGridUL, offset); % Можно добавить
    end

    drawnow limitrate;
    pause(0.05); % Небольшая пауза
end
fprintf('Симуляция завершена.\n');

%% Построение графиков истории (Опционально)
if t == simTime % Строим графики, только если симуляция завершилась полностью
    fprintf('Построение графиков истории...\n');

    % График SINR UL (средний по RB)
    hFigSinr = figure('Name', 'История SINR UL (средний)');
    plot(timeVector, sinrHistoryUL', 'LineWidth', 1.5);
    xlabel('Время (сек / шаг симуляции)');
    ylabel('Средний SINR UL (дБ)');
    title('Динамика среднего SINR UL для каждого UE');
    legend(arrayfun(@(x) sprintf('UE %d', x), 1:numUEs, 'UniformOutput', false), 'Location', 'best');
    grid on;

    % График Path Loss
    hFigPL = figure('Name', 'История Path Loss');
    plot(timeVector, pathLossHistory', 'LineWidth', 1.5);
    xlabel('Время (сек / шаг симуляции)');
    ylabel('Path Loss + Shadowing (дБ)'); % Теперь включает и тень
    title('Динамика Path Loss + Shadowing для каждого UE');
    legend(arrayfun(@(x) sprintf('UE %d', x), 1:numUEs, 'UniformOutput', false), 'Location', 'best');
    grid on;

    fprintf('Графики построены.\n');
end

%% --- Локальные Функции ---
function updateAnnotations(hAxes, gridData, valOffset)
% Добавляет текстовые подписи для каждого непустого блока в ресурсной сетке.
% hAxes: Оси графика (например, gca)
% gridData: Матрица данных сетки (RB x TTI)
% valOffset: Смещение значения, которое нужно вычесть для получения ID UE (0 для DL, offset для UL)

    % Удаляем предыдущие аннотации на этих осях
    delete(findall(hAxes, 'Type', 'Text', 'Tag', 'Annotation'));

    [numRB, numTTI] = size(gridData);

    % Определяем видимые пределы осей, чтобы не рисовать текст за пределами
    xLim = get(hAxes, 'XLim');
    yLim = get(hAxes, 'YLim');
    
    % Находим последний видимый столбец/строку (+ небольшой запас)
    firstVisibleTTI = max(1, floor(xLim(1))+1);
    lastVisibleTTI = min(numTTI, ceil(xLim(2)));
    firstVisibleRB = max(1, floor(yLim(1))+1);
    lastVisibleRB = min(numRB, ceil(yLim(2)));

    % Оптимизация: Не рисуем текст, если он слишком мелкий
    % Примерный порог: если ширина одной ячейки < 5 пикселей
    %axesPos = get(hAxes, 'Position'); % Позиция осей в пикселях [left bottom width height]
    %pixelsPerTTI = axesPos(3) / (xLim(2) - xLim(1));
    %if pixelsPerTTI < 5 % Если ячейки слишком узкие, пропускаем аннотации
     %    % fprintf('Слишком мелко для аннотаций\n');
     %    return;
    %end

    for tti = firstVisibleTTI:lastVisibleTTI % Идем только по видимым столбцам
        for rb = firstVisibleRB:lastVisibleRB % Идем только по видимым строкам
            ue_val = gridData(rb, tti);
            if ue_val > 0 % Если ячейка не пустая
                ue_id = ue_val - valOffset; % Получаем реальный ID UE
                if ue_id > 0 % Убедимся что ID положительный
                     str = sprintf('UE%d', ue_id);
                     % Рисуем текст в центре ячейки (tti-0.5, rb) -> (tti, rb) для imagesc
                     text(hAxes, tti, rb, str, ...
                          'Color','k', ...
                          'FontSize', 7, ... % Уменьшим шрифт
                          'FontWeight', 'bold',...
                          'HorizontalAlignment', 'center', ...
                          'VerticalAlignment', 'middle',...
                          'Clipping', 'on', ... % Обрезать текст, выходящий за пределы
                          'Tag','Annotation'); % Добавляем тег для последующего удаления
                end
            end
        end
    end
end
% --- Определения Классов должны быть ниже или в отдельных файлах ---
% (SystemLevelSimulator и CustomScheduler)