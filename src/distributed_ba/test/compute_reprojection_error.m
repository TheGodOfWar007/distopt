%% Compute the reprojection error for a specific graph
% 04.12.2020 - M.Karrer
clear all; clc;

%% Input data
frame_file = '/home/karrerm/Documents/Debug/One_Way/frames_opt.csv';
map_point_file = '/home/karrerm/Documents/Debug/One_Way/map_points_opt.csv';
observation_file = '/home/karrerm/Documents/Debug/One_Way/Graph_0/observations.csv';

%% Read the data
T = readtable(frame_file);
frame_data = table2array([T(:,1:9), T(:,13:14)]);
map_point_data = csvread(map_point_file);
obs_data = csvread(observation_file);

residual = zeros(2, size(obs_data, 1));
errors = zeros(size(obs_data, 1), 1);
for i = 1:size(obs_data,1)
    frame_idx = find(frame_data(:,1) == obs_data(i, 1));
    map_point_idx = find(map_point_data(:,1) == obs_data(i, 2));
    if (isempty(frame_idx)) 
        fprintf('Could not find frame with id %d\n', obs_data(i, 1));
        continue;
    end
    if (isempty(map_point_idx)) 
        fprintf('Could not find map point with id %d\n', obs_data(i, 2));
        continue;
    end
    T_W_C = eye(4, 4);
    T_W_C(1:3, end) = [frame_data(frame_idx, 2:4)'];
    T_W_C(1:3,1:3) = quat2rotm([frame_data(frame_idx, 8), frame_data(frame_idx, 5:7)]);
    f = frame_data(frame_idx, 9);
    k1 = frame_data(frame_idx, 10);
    k2 = frame_data(frame_idx, 11);
    point_in_W = map_point_data(map_point_idx, 2:4)';
    proj = projectPoint(f, k1, k2, point_in_W, T_W_C);
    residual(:, i) = proj - obs_data(i, 3:4)';
    errors(i) = norm(residual(:, i));
    if (errors(i) > 1e4)
        errors(i) = 0;
        residual(:, i) = zeros(2,1);
    end
end

