%% apply_under_to_day_images_from_night_labels
% Читает night_labels.csv (path,blur,under,over,night,dist_type,dist_level,ref)
% Если under==1 и night==0 -> применяет искажение dist_type=17 (darken)
% и ПЕРЕЗАПИСЫВАЕТ изображение ПО ТОМУ ЖЕ ПУТИ (имя не меняется).
%
% ВАЖНО:
% 1) Этот скрипт изменяет исходные файлы. Для подстраховки можно включить BACKUP.
% 2) dist_level выбирается случайно из [4 5] (как у тебя в KADIS).
% 3) dist_type/dist_level в таблице обновляются и CSV перезаписывается.

%% setup
clear; clc;
addpath(genpath('code_imdistort'));

INPUT_CSV  = 'night_labels.csv';
OUTPUT_CSV = 'night_labels.csv';  % перезаписать этот же файл

% Подстраховка: сделать копию файла перед перезаписью (1 = да, 0 = нет)
BACKUP_CSV = 1;
BACKUP_DIR = 'backup_before_under';

% Для 17 типа (darken) берём уровни как в проекте
allowed_levels_17 = [4 5];

% Качество JPEG при перезаписи (если файл jpg/jpeg)
JPEG_QUALITY = 95;

%% read labels
tb = readtable(INPUT_CSV);

% базовая проверка колонок
needCols = {'path','blur','under','over','night','dist_type','dist_level','ref'};
for c = 1:numel(needCols)
    if ~ismember(needCols{c}, tb.Properties.VariableNames)
        error('В CSV нет обязательной колонки: %s', needCols{c});
    end
end

if BACKUP_CSV
    if ~exist(BACKUP_DIR, 'dir'), mkdir(BACKUP_DIR); end
    copyfile(INPUT_CSV, fullfile(BACKUP_DIR, ['night_labels_backup_' datestr(now,'yyyymmdd_HHMMSS') '.csv']));
end

%% find rows to process: under=1 and night=0
mask = (tb.under == 1) & (tb.night == 0);

idx = find(mask);
fprintf('[INFO] Найдено строк для обработки (under=1 & night=0): %d\n', numel(idx));

if isempty(idx)
    fprintf('[INFO] Нечего делать. Выход.\n');
    return;
end

%% process each image
processed = 0;
skipped_missing = 0;
skipped_readerr  = 0;

for k = 1:numel(idx)
    i = idx(k);

    img_path = tb.path{i};

    if ~isfile(img_path)
        skipped_missing = skipped_missing + 1;
        fprintf('[WARN] Файл не найден: %s\n', img_path);
        continue;
    end

    try
        im = imread(img_path);
    catch
        skipped_readerr = skipped_readerr + 1;
        fprintf('[WARN] Не удалось прочитать: %s\n', img_path);
        continue;
    end

    % выбираем уровень затемнения
    dist_type  = 17;
    dist_level = allowed_levels_17(randi(numel(allowed_levels_17)));

    % применяем искажение
    dist_im = imdist_generator(im, dist_type, dist_level);

    % перезаписываем БЕЗ изменения имени
    [~,~,ext] = fileparts(img_path);
    ext = lower(ext);

    try
        if strcmp(ext, '.jpg') || strcmp(ext, '.jpeg')
            imwrite(dist_im, img_path, 'Quality', JPEG_QUALITY);
        else
            % Для png/bmp/webp и т.п. пишем без параметра Quality
            imwrite(dist_im, img_path);
        end
    catch
        fprintf('[WARN] Не удалось записать: %s\n', img_path);
        continue;
    end

    % обновляем поля в таблице (логически: это "синтетический under")
    tb.dist_type(i)  = dist_type;
    tb.dist_level(i) = dist_level;

    % blur/over оставляем 0, under уже 1, night уже 0 (как и было)
    tb.blur(i) = 0;
    tb.over(i) = 0;

    processed = processed + 1;

    if mod(processed, 200) == 0
        fprintf('[INFO] Обработано %d / %d\n', processed, numel(idx));
    end
end

%% write updated CSV back
writetable(tb, OUTPUT_CSV);
fprintf('[OK] Готово.\n');
fprintf('  Перезаписано изображений: %d\n', processed);
fprintf('  Пропущено (нет файла): %d\n', skipped_missing);
fprintf('  Пропущено (ошибка чтения): %d\n', skipped_readerr);
fprintf('  CSV обновлён: %s\n', OUTPUT_CSV);