classdef SystemLevelSimulator < handle
    properties
        % Основные параметры
        NumUEs
        Bandwidth
        SCS
        NumRBs
        SimTime
        CarrierFrequency % Гц
        GNBTxPower       % дБм
        UETxPower        % дБм

        % Состояние UE
        UELocations     % Позиции UE [NumUEs x 2]
        Velocity        % Текущая скорость каждого UE [Vx, Vy] (м/с) - для GaussMarkov

        % Параметры среды
        gNBLocation     % Позиция базовой станции [x, y]
        MapSize = [500, 500] % Размер карты [maxX, maxY]
        AntennaHeightUE = 1.5;   % м
        AntennaHeightGNB = 35;   % м

        % Параметры Мобильности
        MobilityModelType = 'GaussMarkov'; % 'RandomWaypoint' или 'GaussMarkov'
        DeltaTGM = 1; % Временной шаг для обновления мобильности (сек)
        % Random Waypoint specific
        DestinationRWP  % Целевые точки для UE [NumUEs x 2]
        SpeedRWP = 8;   % Скорость для RWP (м/с)
        % Gauss-Markov specific
        AlphaGM = 0.8;   % Коэффициент "памяти" (0 <= alpha < 1)
        SigmaSpeedGM = 1;% Стандартное отклонение изменения скорости (м/с)
        SigmaDirGM = pi/6;% Стандартное отклонение изменения направления (рад)
        MeanSpeedGM = 5; % Средняя скорость (м/с)

        % Параметры Канала
        ChannelModelType = 'Custom'; % 'Custom', 'LTE_ETU', 'LTE_EVA', 'LTE_EPA'
        % Общие параметры канала
        SigmaShadow = 8.0;   % дБ
        UseShadowing = true;
        UseFastFading = true;
        CustomFastFadingVar = 3.0; % Примерная вариация дБ для быстр. затух.
        
        % Параметры для LTE/5G моделей
        ChannelSamplingRate
        ChannelModelsLTE     % Cell array для хранения объектов lteTDLChannel
        ChannelModels5G      % Cell array для хранения объектов nrTDLChannel
        
        % Состояние Канала (результаты updateChannelConditions)
        ChannelInfo % Структура с результатами (RSRP, CQI и т.д.)

        % Трафик-модель
        TrafficDL       % Трафик для DL (пакетов/сек)
        TrafficUL       % Трафик для UL (пакетов/сек)
        lambdaDL = 5;   % Среднее число пакетов в DL
        lambdaUL = 3;   % Среднее число пакетов в UL
        TrafficBuffer   % Упрощенный буфер DL

    end

    methods
        % =================================================================
        % --- Конструктор ---
        % =================================================================
        function obj = SystemLevelSimulator(numUEs, bw, scs, simTime, fc, mobilityModel, channelModel, gnbTxPower, ueTxPower)
            obj.NumUEs = numUEs;
            obj.Bandwidth = bw;
            obj.SCS = scs;
            obj.SimTime = simTime;
            obj.CarrierFrequency = fc;
            obj.GNBTxPower = gnbTxPower;
            obj.UETxPower = ueTxPower;
            obj.NumRBs = floor(bw / (12 * scs));
            obj.ChannelSamplingRate = bw; % Упрощенно

            obj.gNBLocation = [obj.MapSize(1)/2, obj.MapSize(2)/2]; % Центр карты

            % Установка моделей
            obj.MobilityModelType = mobilityModel;
            obj.ChannelModelType = channelModel;
            % Инициализация UE и мобильности
            obj.initializeUE();
            obj.initializeMobility(); % Инициализация специфичных параметров мобильности
            obj.initializeChannelModels(); % Инициализация объектов канала (если нужны)

            % Инициализация трафика
            obj.TrafficBuffer = zeros(numUEs,1);
            obj.updateTraffic(); % задаём начальные значения трафика

            % Пустая структура для ChannelInfo
            obj.ChannelInfo = struct();


            fprintf('Симулятор создан: %d UE, BW=%.1f MHz, SCS=%d kHz, Mob=%s, Chan=%s\n', ...
                    numUEs, bw/1e6, scs/1e3, mobilityModel, channelModel);
                if contains(obj.ChannelModelType,'LTE','IgnoreCase',true) || contains(obj.ChannelModelType,'5G','IgnoreCase',true)
                    fprintf('  (Используется PL/Sh по 3GPP, FF по %s модели)\n', obj.ChannelModelType);
                elseif strcmpi(obj.ChannelModelType,'Custom')
                    fprintf('  (Используется PL/Sh по 3GPP UMa NLOS, FF по Custom модели)\n');
                 end
           end
        
        % =================================================================
        % --- Методы Инициализации ---
        % =================================================================
        function initializeUE(obj)
            obj.UELocations = rand(obj.NumUEs, 2) .* obj.MapSize;
            obj.Velocity = zeros(obj.NumUEs, 2); % Начальная скорость 0
        end

        function initializeMobility(obj)
            if strcmpi(obj.MobilityModelType, 'RandomWaypoint')
                obj.DestinationRWP = rand(obj.NumUEs, 2) .* obj.MapSize;
                fprintf('Инициализирована мобильность RandomWaypoint (Скорость=%.1f м/с)\n', obj.SpeedRWP);
            elseif strcmpi(obj.MobilityModelType, 'GaussMarkov')
                initial_angle = rand(obj.NumUEs, 1) * 2 * pi;
                initial_speed_magnitude = max(0, obj.MeanSpeedGM + obj.SigmaSpeedGM * randn(obj.NumUEs, 1));
                obj.Velocity = [initial_speed_magnitude .* cos(initial_angle), initial_speed_magnitude .* sin(initial_angle)];
                fprintf('Инициализирована мобильность Gauss-Markov (Alpha=%.2f, MeanSpeed=%.1f м/с)\n', obj.AlphaGM, obj.MeanSpeedGM);
            else
                error('Неизвестный тип модели мобильности: %s', obj.MobilityModelType);
            end
        end

         function initializeChannelModels(obj)
            % Сброс массивов моделей
            obj.ChannelModelsLTE = [];
            obj.ChannelModels5G = [];

            % Установка SigmaShadow в зависимости от сценария (пример)
            if contains(obj.ChannelModelType, 'UMa', 'IgnoreCase', true) || contains(obj.ChannelModelType, 'ETU', 'IgnoreCase', true)
                 obj.SigmaShadow = 8.0; % UMa NLOS
            elseif contains(obj.ChannelModelType, 'RMa', 'IgnoreCase', true) || contains(obj.ChannelModelType, 'EVA', 'IgnoreCase', true) || contains(obj.ChannelModelType, 'EPA', 'IgnoreCase', true)
                 obj.SigmaShadow = 6.0; % RMa NLOS
            else % Custom или неизвестный
                 obj.SigmaShadow = 8.0; % По умолчанию как для UMa NLOS
            end

            % Инициализация для LTE моделей
            if contains(obj.ChannelModelType, 'LTE', 'IgnoreCase', true)
                [toolboxAvailable, errmsg] = license('checkout', 'LTE_Toolbox');
                if ~toolboxAvailable
                     warning('Не удалось получить лицензию LTE Toolbox: "%s". Проверьте статус лицензии. Используется Custom модель канала.', errmsg);
                     obj.ChannelModelType = 'Custom'; % Откат к Custom
                     fprintf('Откат к модели канала: Custom (3GPP PL/Sh + Custom FF)\n');
                     return; % Выходим, дальше инициализировать нечего
                else
                     license('checkin', 'LTE_Toolbox'); % Возвращаем лицензию
                     fprintf('Лицензия LTE Toolbox доступна.\n');
                end

                fprintf('Инициализация LTE TDL канала (%s)...\n', obj.ChannelModelType);
                obj.ChannelModelsLTE = cell(obj.NumUEs, 1);

                  % Определение профиля LTE
                 if contains(obj.ChannelModelType,'ETU','IgnoreCase',true)
                      lteDelayProfile = 'ETU';
                      % Можно задать SigmaShadow3GPP здесь, если нужно
                 elseif contains(obj.ChannelModelType,'EVA','IgnoreCase',true)
                      lteDelayProfile = 'EVA';
                 elseif contains(obj.ChannelModelType,'EPA','IgnoreCase',true)
                      lteDelayProfile = 'EPA';
                 else
                     warning('Неизвестный LTE профиль в ChannelModelType: %s. Используется ETU.', obj.ChannelModelType);
                     lteDelayProfile = 'ETU';
                 end
                 fprintf('  LTE Профиль: %s, Тень 3GPP: %.1f dB\n', ...
                         lteDelayProfile, obj.SigmaShadow3GPP);

                for ue = 1:obj.NumUEs
                    channel = lteTDLChannel;
                    channel.DelayProfile = lteDelayProfile;
                    channel.SampleRate = obj.ChannelSamplingRate;
                    channel.MaximumDopplerShift = 5; % Начальное значение, будет обновляться
                    channel.NumTransmitAntennas = 1; % Упрощенно SISO
                    channel.NumReceiveAntennas = 1;  % Упрощенно SISO
                    channel.CarrierFrequency = obj.CarrierFrequency; % Задаем несущую
                    % Можно добавить другие параметры TDL по необходимости
                    obj.ChannelModelsLTE{ue} = channel;
                end

                % Инициализация для 5G моделей
            elseif contains(obj.ChannelModelType, '5G', 'IgnoreCase', true)
                [toolboxAvailable, errmsg] = license('checkout', '5G_Toolbox');
                 if ~toolboxAvailable
                     warning('Не удалось получить лицензию 5G Toolbox: "%s". Проверьте статус лицензии. Используется Custom модель канала.', errmsg);
                     obj.ChannelModelType = 'Custom'; % Откат к Custom
                     fprintf('Откат к модели канала: Custom (3GPP PL/Sh + Custom FF)\n');
                     return; % Выходим
                 else
                      license('checkin', '5G_Toolbox');
                      fprintf('Лицензия 5G Toolbox доступна.\n');
                 end

                fprintf('Инициализация 5G TDL канала (%s)...\n', obj.ChannelModelType);
                obj.ChannelModels5G = cell(obj.NumUEs, 1);
                 % Определяем параметры TDL на основе имени модели
                 if contains(obj.ChannelModelType,'UMa','IgnoreCase',true)
                      nrDelayProfile = 'TDL-C'; nrDelaySpread = 100e-9;
                 elseif contains(obj.ChannelModelType,'RMa','IgnoreCase',true)
                      nrDelayProfile = 'TDL-A'; nrDelaySpread = 30e-9;
                 else
                      nrDelayProfile = 'TDL-C'; nrDelaySpread = 100e-9; % По умолчанию UMa
                 end
                 fprintf('  5G Профиль: %s, Разброс задержек: %.1f ns, SigmaShadow=%.1f dB\n', ...
                         nrDelayProfile, nrDelaySpread*1e9, obj.SigmaShadow);

                for ue = 1:obj.NumUEs
                    channel = nrTDLChannel;
                    channel.DelayProfile = nrDelayProfile;
                    channel.DelaySpread = nrDelaySpread;
                    channel.SampleRate = obj.ChannelSamplingRate;
                    channel.MaximumDopplerShift = 5; % Начальное значение
                    channel.NumTransmitAntennas = 1;
                    channel.NumReceiveAntennas = 1;
                    % channel.CarrierFrequency = obj.CarrierFrequency; % У nrTDL нет такого свойства напрямую
                    obj.ChannelModels5G{ue} = channel;
                end

            % Для Custom модели специальная инициализация объектов не нужна
            elseif strcmpi(obj.ChannelModelType, 'Custom')
                 fprintf('Инициализация улучшенной Custom модели канала (3GPP PL/Sh + Custom FF).\n');
                 fprintf('  SigmaShadow=%.1f dB, CustomFastFadingVar=%.1f dB\n', obj.SigmaShadow, obj.CustomFastFadingVar);
            else
                 error('Неподдерживаемый ChannelModelType: %s', obj.ChannelModelType);
            end
         end

        % =================================================================
        % --- Методы Обновления ---
        % =================================================================
        function updateNetwork(obj, t, scheduler)
            obj.updateMobility();
            obj.updateTraffic();
            % Передаем предыдущие распределения из планировщика
            prevULAlloc = scheduler.PreviousULAllocation;
            prevDLAlloc = scheduler.PreviousDLAllocation; % Пока не используется
            obj.updateChannelConditions(prevULAlloc, prevDLAlloc);
        end

        % --- Обновление Мобильности ---
        function updateMobility(obj)
             if strcmpi(obj.MobilityModelType, 'RandomWaypoint')
                obj.updateMobilityRandomWaypoint();
            elseif strcmpi(obj.MobilityModelType, 'GaussMarkov')
                obj.updateMobilityGaussMarkov();
            end
             % Обновление Доплера для канальных моделей
             if obj.UseFastFading
                 currentSpeeds = vecnorm(obj.Velocity, 2, 2);
                 maxDoppler = currentSpeeds * (obj.CarrierFrequency / physconst('LightSpeed'));
                 maxDoppler = max(5, maxDoppler); % Мин. значение 5 Гц

                 if ~isempty(obj.ChannelModelsLTE) % Обновляем LTE модели
                     for ue = 1:obj.NumUEs
                        obj.ChannelModelsLTE{ue}.MaximumDopplerShift = maxDoppler(ue);
                     end
                 end
                 if ~isempty(obj.ChannelModels5G) % Обновляем 5G модели
                      for ue = 1:obj.NumUEs
                         obj.ChannelModels5G{ue}.MaximumDopplerShift = maxDoppler(ue);
                      end
                 end
             end
        end

        function updateMobilityRandomWaypoint(obj)
             for ue = 1:obj.NumUEs
                currentPos = obj.UELocations(ue,:);
                dest = obj.DestinationRWP(ue,:);
                direction = dest - currentPos;
                dist = norm(direction);
                stepSize = obj.SpeedRWP * obj.DeltaTGM;

                if dist < stepSize
                    obj.UELocations(ue,:) = dest;
                    obj.DestinationRWP(ue,:) = rand(1,2).* obj.MapSize;
                    % Остановка -> новое направление (случайная скорость/направление не для RWP)
                    obj.Velocity(ue,:) = [0, 0]; % Остановка в RWP не моделируется явно, но можно обнулить V
                else
                    moveVec = (direction/dist) * stepSize;
                    obj.UELocations(ue,:) = currentPos + moveVec;
                    obj.Velocity(ue,:) = moveVec / obj.DeltaTGM; % Сохраняем скорость для Доплера
                end
                % Проверка границ
                [obj.UELocations(ue,:), obj.Velocity(ue,:)] = obj.handleBoundaries(obj.UELocations(ue,:), obj.Velocity(ue,:));
            end
        end

        function updateMobilityGaussMarkov(obj)
            for ue = 1:obj.NumUEs
                current_speed_mag = norm(obj.Velocity(ue,:));
                current_angle = atan2(obj.Velocity(ue,2), obj.Velocity(ue,1));
                if isnan(current_angle), current_angle = 0; end % Если скорость 0

                speed_noise = obj.SigmaSpeedGM * randn();
                new_speed_mag = obj.AlphaGM * current_speed_mag + (1 - obj.AlphaGM) * obj.MeanSpeedGM + sqrt(1 - obj.AlphaGM^2) * speed_noise;
                new_speed_mag = max(0, new_speed_mag);

                angle_noise = obj.SigmaDirGM * randn();
                % Среднее направление сохраняется (можно заменить на случайное, если нужно)
                mean_angle = current_angle;
                new_angle = obj.AlphaGM * current_angle + (1 - obj.AlphaGM) * mean_angle + sqrt(1 - obj.AlphaGM^2) * angle_noise;

                obj.Velocity(ue,:) = [new_speed_mag * cos(new_angle), new_speed_mag * sin(new_angle)];
                obj.UELocations(ue,:) = obj.UELocations(ue,:) + obj.Velocity(ue,:) * obj.DeltaTGM;
                % Проверка границ
                [obj.UELocations(ue,:), obj.Velocity(ue,:)] = obj.handleBoundaries(obj.UELocations(ue,:), obj.Velocity(ue,:));
            end
        end

        function [pos, vel] = handleBoundaries(obj, pos, vel)
             % Обработка границ карты (отражение)
             if pos(1) <= 0
                 pos(1) = 0;
                 vel(1) = abs(vel(1)); % Направляем внутрь
             elseif pos(1) >= obj.MapSize(1)
                 pos(1) = obj.MapSize(1);
                 vel(1) = -abs(vel(1)); % Направляем внутрь
             end
             if pos(2) <= 0
                 pos(2) = 0;
                 vel(2) = abs(vel(2));
             elseif pos(2) >= obj.MapSize(2)
                 pos(2) = obj.MapSize(2);
                 vel(2) = -abs(vel(2));
             end
        end

        % --- Обновление Трафика ---
        function updateTraffic(obj)
            obj.TrafficDL = poissrnd(obj.lambdaDL, obj.NumUEs, 1);
            obj.TrafficUL = poissrnd(obj.lambdaUL, obj.NumUEs, 1);
            obj.TrafficBuffer = obj.TrafficBuffer + obj.TrafficDL;
            % Упрощенная модель передачи (можно улучшить)
            transmitted = min(obj.TrafficBuffer, randi([1, 5], obj.NumUEs, 1));
            obj.TrafficBuffer = max(0, obj.TrafficBuffer - transmitted);
        end

        % =================================================================
        % --- Обновление Канала ---
        % =================================================================
        function updateChannelConditions(obj, previousULAllocation, ~) % prevDLAlloc пока не используется

            distances = vecnorm(obj.UELocations - obj.gNBLocation, 2, 2);
            distances = max(10, distances); % Мин. дистанция для избежания проблем с log10

            % --- Расчет Path Loss + Shadowing (по 3GPP TR 38.901) ---
            % Выбираем формулу на основе ТИПА канала (UMa/RMa)
            % Используем эти расчеты для Custom, LTE, 5G моделей
            h_ue = obj.AntennaHeightUE;
            h_bs = obj.AntennaHeightGNB;
            d_2d = distances;
            d_3d = sqrt(d_2d.^2 + (h_bs - h_ue)^2);
            fc_GHz = obj.CarrierFrequency / 1e9;
            PL_base = 28.0 + 22*log10(d_3d) + 20*log10(fc_GHz);
            pathLoss_dB = zeros(obj.NumUEs, 1);
            
            isUMaScenario = contains(obj.ChannelModelType, {'UMa', 'ETU', 'Custom'}, 'IgnoreCase', true); % Считаем Custom как UMa по умолчанию для PL/Sh
            isRMaScenario = contains(obj.ChannelModelType, {'RMa', 'EVA', 'EPA'}, 'IgnoreCase', true);
            
            if isUMaScenario
                % Формула UMa NLOS из TR 38.901
                PL_UMa_NLOS_part1 = 32.4 + 20*log10(fc_GHz) + 30.9*log10(d_3d);
                PL_UMa_NLOS_part2 = PL_base; % TR 38.901 Table B.1.2.1-1 использует max(..., LOS_PL), где LOS_PL=PL_base
                PL_UMa_NLOS_dB = max(PL_UMa_NLOS_part1, PL_UMa_NLOS_part2);
                pathLoss_dB = -PL_UMa_NLOS_dB;
                currentSigmaShadow = 8.0; % Для UMa NLOS
            elseif isRMaScenario
                 % Формула RMa NLOS из TR 38.901
                 PL_RMa_NLOS_part1 = 22.4 + 35.3*log10(d_3d) + 21.3*log10(fc_GHz) - 0.3*(h_ue-1.5);
                 PL_RMa_NLOS_part2 = PL_base;
                 PL_RMa_NLOS_dB = max(PL_RMa_NLOS_part1, PL_RMa_NLOS_part2);
                 pathLoss_dB = -PL_RMa_NLOS_dB;
                 currentSigmaShadow = 6.0; % Для RMa NLOS
            else % Неизвестный тип - используем UMa по умолчанию
                 warning('Неизвестный сценарий для PL/Sh в ChannelModelType: %s. Используется UMa NLOS.', obj.ChannelModelType);
                 PL_UMa_NLOS_part1 = 32.4 + 20*log10(fc_GHz) + 30.9*log10(d_3d);
                 PL_UMa_NLOS_part2 = PL_base;
                 PL_UMa_NLOS_dB = max(PL_UMa_NLOS_part1, PL_UMa_NLOS_part2);
                 pathLoss_dB = -PL_UMa_NLOS_dB;
                 currentSigmaShadow = 8.0;
            end        
            
            shadowing_dB = 0;
            if obj.UseShadowing
                shadowing_dB = currentSigmaShadow * randn(obj.NumUEs, 1);
            end
            pathLossShadow_dB = pathLoss_dB + shadowing_dB;

            % --- Расчет Быстрых Замираний (Fast Fading) ---
            fastFadingGain_UL_linear = ones(obj.NumUEs, 1);
            fastFadingGain_DL_linear = ones(obj.NumUEs, 1);

            if obj.UseFastFading
                if strcmpi(obj.ChannelModelType, 'Custom')
                    fastFading_UL_dB = obj.CustomFastFadingVar * randn(obj.NumUEs, 1);
                    fastFading_DL_dB = obj.CustomFastFadingVar * randn(obj.NumUEs, 1);
                    fastFadingGain_UL_linear = 10.^(fastFading_UL_dB / 10);
                    fastFadingGain_DL_linear = 10.^(fastFading_DL_dB / 10);
                
                 elseif contains(obj.ChannelModelType, 'LTE', 'IgnoreCase', true) && ~isempty(obj.ChannelModelsLTE)
                    for ue = 1:obj.NumUEs
                        channel = obj.ChannelModelsLTE{ue};
                        % Упрощенный расчет среднего усиления
                        impulse = [1; zeros(channel.MaximumChannelDelay+10,1)]; % Увеличил длину импульса
                         try % Добавляем try-catch на случай ошибок в канале
                            [~, pathGains] = channel(impulse);
                            avgPowerGain = mean(sum(abs(pathGains).^2, 1));
                            if isnan(avgPowerGain) || avgPowerGain <= 0, avgPowerGain = 1; end
                            fastFadingGain_UL_linear(ue) = avgPowerGain;
                            fastFadingGain_DL_linear(ue) = avgPowerGain; % Считаем одинаковым для DL/UL
                         catch ME
                            warning('Ошибка при получении усиления из LTE канала для UE %d: %s. Используется усиление 1.', ue, ME.message);
                            fastFadingGain_UL_linear(ue) = 1;
                            fastFadingGain_DL_linear(ue) = 1;
                         end
                    end

                elseif contains(obj.ChannelModelType, '5G', 'IgnoreCase', true) && ~isempty(obj.ChannelModels5G)
                     for ue = 1:obj.NumUEs
                        channel = obj.ChannelModels5G{ue};
                        % Упрощенный расчет среднего усиления для nrTDLChannel
                         try
                            info = info(channel); % Получаем информацию о канале
                            pathGains = info.PathGains; % Коэф. путей
                            avgPowerGain = mean(sum(abs(pathGains).^2, 1));
                            if isnan(avgPowerGain) || avgPowerGain <= 0, avgPowerGain = 1; end
                            fastFadingGain_UL_linear(ue) = avgPowerGain;
                            fastFadingGain_DL_linear(ue) = avgPowerGain;
                         catch ME
                             warning('Ошибка при получении усиления из 5G канала для UE %d: %s. Используется усиление 1.', ue, ME.message);
                             fastFadingGain_UL_linear(ue) = 1;
                             fastFadingGain_DL_linear(ue) = 1;
                         end
                     end
                % else: Если модель LTE/5G выбрана, но объекты не создались (лицензия/ошибка), усиление останется 1
                end
            end % end UseFastFading
            
            % --- Итоговое Усиление Канала ---
            channelGainUL_linear = 10.^(pathLossShadow_dB / 10) .* fastFadingGain_UL_linear;
            channelGainDL_linear = 10.^(pathLossShadow_dB / 10) .* fastFadingGain_DL_linear;

            % --- Расчет Принятой Мощности и RSRP (на gNB от UE) ---
            txPowerUE_mW = 10^((obj.UETxPower - 30) / 10);
            receivedPowerUE_mW = txPowerUE_mW * channelGainUL_linear;
            RSRP_UL_dBm = 10*log10(max(1e-20, receivedPowerUE_mW)) + 30; % RSRP на gNB

             % --- Расчет Принятой Мощности и RSRP (на UE от gNB) ---
             txPowerGNB_mW = 10^((obj.GNBTxPower - 30) / 10);
             receivedPowerGNB_mW = txPowerGNB_mW * channelGainDL_linear;
             RSRP_DL_dBm = 10*log10(max(1e-20, receivedPowerGNB_mW)) + 30; % RSRP на UE

            % --- Расчет Шума ---
            noiseFloor_dBm = -174 + 10 * log10(obj.Bandwidth);
            noisePower_mW = 10^((noiseFloor_dBm - 30) / 10);
            noisePerRB_mW = noisePower_mW / obj.NumRBs;

            % --- Расчет Интерференции (UL, на gNB) ---
            interferencePowerPerRB_mW = zeros(1, obj.NumRBs);
            for rb = 1:obj.NumRBs
             ue_id_on_rb_prev = previousULAllocation(rb);

             if ue_id_on_rb_prev > 0 && ue_id_on_rb_prev <= obj.NumUEs
                   interferencePowerPerRB_mW(rb) = receivedPowerUE_mW(ue_id_on_rb_prev);
             end
            end

            % --- Расчет SINR и CQI (UL, на gNB) ---
        SINR_UL_PerRB = zeros(obj.NumUEs, obj.NumRBs);
        for ue = 1:obj.NumUEs % Цикл по UE, для которого считаем SINR
            signalPower_mW = receivedPowerUE_mW(ue); % Мощность сигнала от интересующего нас UE

            for rb = 1:obj.NumRBs % Цикл по всем RB
                % Интерференция на этом RB - это вся мощность, что была на нем в прошлом слоте,
                % ЗА ИСКЛЮЧЕНИЕМ мощности самого интересующего нас UE 'ue',
                % ЕСЛИ именно он и передавал на этом RB.

                totalReceivedPowerOnRB_lastSlot = interferencePowerPerRB_mW(rb); % Общая мощность на RB в t-1

                ue_id_on_rb_prev = previousULAllocation(rb); % Кто был на этом RB в t-1?

                if ue_id_on_rb_prev == ue
                    % Если интересующий нас UE сам передавал на этом RB,
                    % то для него интерференция на этом RB равна 0 (идеально).
                    % В реальных системах может быть межканальная интерференция, но здесь упрощаем.
                    interferenceExcludingSelf_mW = 0;
                else
                    % Если на этом RB передавал ДРУГОЙ UE (или никто),
                    % то вся принятая мощность на этом RB является интерференцией
                    % для нашего UE 'ue'.
                    interferenceExcludingSelf_mW = totalReceivedPowerOnRB_lastSlot;
                end
                noiseAndInterference_mW = noisePerRB_mW + interferenceExcludingSelf_mW;
                % Избегаем деления на ноль или очень малые значения
                if noiseAndInterference_mW < 1e-20
                    noiseAndInterference_mW = 1e-20;
                end
                SINR_UL_PerRB(ue, rb) = signalPower_mW / noiseAndInterference_mW;
            end
        end
        
            SINR_UL_PerRB_dB = 10*log10(max(1e-20,SINR_UL_PerRB)); % Защита от log10(0)
            SINR_UL_PerRB_dB = max(-20, min(30, SINR_UL_PerRB_dB)); % Ограничение

            % --- Расчет CQI UL per RB ---
            CQI_UL_PerRB = zeros(obj.NumUEs, obj.NumRBs);
            % Уровни SINR для CQI (Примерные, можно уточнить по стандартам)
            cqi_sinr_thresholds_dB = [-Inf, -6.5, -4.5, -2.5, -0.5, 1.5, 3.5, 5.5, 7.5, 9.8, 11.8, 13.8, 15.8, 18.0, 20.0]; % 15 порогов -> 15 CQI
            for ue = 1:obj.NumUEs
                % +1 т.к. find возвращает индекс порога <= SINR, а нам нужен номер CQI
                cqi_indices = sum(bsxfun(@ge, SINR_UL_PerRB_dB(ue,:)', cqi_sinr_thresholds_dB), 2);
                CQI_UL_PerRB(ue, :) = max(1, cqi_indices'); % Используем max(1,...) чтобы избежать CQI 0
            end
            CQI_UL_PerRB = min(CQI_UL_PerRB, 15); % Ограничиваем сверху

            % --- Расчет SINR и CQI (DL, на UE) - ЗАГЛУШКА ---
            % В этой версии интерференция DL не моделируется.
            % SINR = Signal / Noise
            SINR_DL_PerRB = zeros(obj.NumUEs, obj.NumRBs);
             for ue = 1:obj.NumUEs
                 signalPower_mW = receivedPowerGNB_mW(ue); % Мощность от gNB
                 % Распределяем равномерно по RB (упрощение)
                 signalPerRB_mW = signalPower_mW / obj.NumRBs;
                  % Шум UE считаем таким же как шум gNB (упрощение)
                 noiseAndInterference_mW = noisePerRB_mW; % Только шум
                if noiseAndInterference_mW < 1e-20,noiseAndInterference_mW = 1e-20; 
                end
                 SINR_DL_PerRB(ue, :) = signalPerRB_mW / noiseAndInterference_mW;
             end
            SINR_DL_PerRB_dB = 10*log10(max(1e-20,SINR_DL_PerRB));
            SINR_DL_PerRB_dB = max(-20, min(30, SINR_DL_PerRB_dB)); % Ограничение

            CQI_DL_PerRB = zeros(obj.NumUEs, obj.NumRBs);
             for ue = 1:obj.NumUEs
                 cqi_indices = sum(bsxfun(@ge, SINR_DL_PerRB_dB(ue,:)', cqi_sinr_thresholds_dB), 2);
                 CQI_DL_PerRB(ue, :) = max(1, cqi_indices');
             end
             CQI_DL_PerRB = min(CQI_DL_PerRB, 15);

            % --- Формирование ChannelInfo ---
            obj.ChannelInfo = struct(...
                'RSRP', RSRP_DL_dBm, ... % RSRP на UE от gNB
                'PathLoss_dB', pathLossShadow_dB, ... % PL+Shadow для истории
                'CQI_DL', round(mean(CQI_DL_PerRB, 2)), ... % Средний CQI DL
                'CQI_UL', round(mean(CQI_UL_PerRB, 2)), ... % Средний CQI UL
                'CQI_DL_PerRB', CQI_DL_PerRB, ... % [NumUEs x NumRBs]
                'CQI_UL_PerRB', CQI_UL_PerRB, ... % [NumUEs x NumRBs]
                'SINR_UL_Avg', mean(SINR_UL_PerRB_dB, 2), ... % Средний SINR UL для истории
                'TrafficDL', obj.TrafficDL, ...
                'TrafficUL', obj.TrafficUL, ...
                'TrafficBuffer', obj.TrafficBuffer ...
            );
        end 

    end % methods
end % classdef