%% setup
clear; clc;
addpath(genpath('code_imdistort'));

fid = fopen('labels.csv','w');
fprintf(fid, 'path,blur,under,over,dist_type,dist_level,ref\n');

%% read the info of pristine images

tb = readtable('kadis700k_ref_imgs.csv');

% --- ОСТАВЛЯЕМ ТОЛЬКО НУЖНЫЕ ИСКАЖЕНИЯ ---
types = [2 3 16 17];            % gaussian/lens/motion blur + brighten + darken
tb = tb(ismember(tb{:,2}, types), :);  % предполагаем, что 2-й столбец = dist_type

tb = table2cell(tb);


%% labels for ref

ref_list = unique(tb(:,1));  % это cell array с именами файлов

for r = 1:numel(ref_list)
    ref_name = ref_list{r};
    ref_path = fullfile('ref_imgs', ref_name);

    % pristine: дефектов нет
    blur = 0; under = 0; over = 0;
    dist_type = 0; dist_level = 0;

    % ref в конце можно тоже писать ref_name (для pristine это одно и то же)
    fprintf(fid, '%s,%d,%d,%d,%d,%d,%s\n', ref_path, blur, under, over, dist_type, dist_level, ref_name);
end


%% generate distorted images in dist_imgs folder

for i = 1:size(tb,1)
    ref_im = imread(fullfile('ref_imgs', tb{i,1}));
    dist_type = tb{i,2};

    if dist_type == 2
        allowed_levels = [2 3 4];
    elseif dist_type == 3
        allowed_levels = [4 5];
    elseif dist_type == 16
        allowed_levels = [4 5];
    elseif dist_type == 17
        allowed_levels = [4 5];
    else
        continue;
    end
    
  %%  for k = 1:numel(keep_levels)
        dist_level = allowed_levels(randi(numel(allowed_levels)));

        dist_im = imdist_generator(ref_im, dist_type, dist_level);

        % имя файла остаётся совместимым: _тип_уровень.bmp
        strs = split(tb{i,1},'.');

        dist_im_name = [strs{1} '_' num2str(dist_type,'%02d') '_' num2str(dist_level,'%02d') '.jpg'];
        out_path = fullfile('dist_imgs', dist_im_name);

        if exist(out_path, 'file')
            continue;
        end
        
        imwrite(dist_im, out_path, 'Quality', 95);

        % вычисляем метки по dist_type
        blur = 0; under = 0; over = 0;
        if dist_type == 2 || dist_type == 3
            blur = 1;
        elseif dist_type == 17
            under = 1;
        elseif dist_type == 16
            over = 1;
        end
       
        fprintf(fid, '%s,%d,%d,%d,%d,%d,%s\n', out_path, blur, under, over, dist_type, dist_level, tb{i,1});

 %%   end
end

fclose(fid);
