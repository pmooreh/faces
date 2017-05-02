
% Sliding window face detection with linear SVM. 
% All code by James Hays, except for pieces of evaluation code from Pascal
% VOC toolkit. Images from CMU+MIT face database, CalTech Web Face
% Database, and SUN scene database.

% Code structure:
% proj4.m <--- You code parts of this
%  + get_positive_features.m  <--- You code this
%  + get_random_negative_features.m  <--- You code this
%   [classifier training]   <--- You code this
%  + report_accuracy.m
%  + run_detector.m  <--- You code this
%    + non_max_supr_bbox.m
%  + evaluate_all_detections.m
%    + VOCap.m
%  + visualize_detections_by_image.m
%  + visualize_detections_by_image_no_gt.m
%  + visualize_detections_by_confidence.m

% Other functions. You don't need to use any of these unless you're trying
% to modify or build a test set:

% Training and Testing data related functions:
% test_scenes/visualize_cmumit_database_landmarks.m
% test_scenes/visualize_cmumit_database_bboxes.m
% test_scenes/cmumit_database_points_to_bboxes.m %This function converts
% from the original MIT+CMU test set landmark points to Pascal VOC
% annotation format (bounding boxes).

% caltech_faces/caltech_database_points_to_crops.m %This function extracts
% training crops from the Caltech Web Face Database. The crops are
% intentionally large to contain most of the head, not just the face. The
% test_scene annotations are likewise scaled to contain most of the head.

% set up paths to VLFeat functions. 
% See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
% This should work on 32 and 64 bit versions of Windows, MacOS, and Linux
close all
clear
run('vlfeat/toolbox/vl_setup')

[~,~,~] = mkdir('visualizations');

data_path = '../data/'; %change if you want to work with a network copy
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces'); %Positive training examples. 36x36 head crops
train_path_pos_sides = fullfile(data_path,'MIT-CBCL-facerec-database/side-views');
train_path_pos_synth = fullfile(data_path,'MIT-CBCL-facerec-database/training-synthetic');

non_face_scn_path = fullfile(data_path, 'train_non_face_scenes'); %We can mine random or hard negatives from here
test_scn_path = fullfile(data_path,'test_scenes/test_jpg_small'); %CMU+MIT test scenes
% test_scn_path = fullfile(data_path,'extra_test_scenes'); %Bonus scenes
label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt'); %the ground truth face locations in the test set

%The faces are 36x36 pixels, which works fine as a template size. You could
%add other fields to this struct if you want to modify HoG default
%parameters such as the number of orientations, but that does not help
%performance in our limited test.
feature_params = struct('template_size', 36, 'hog_cell_size', 6);
feature_params_filter = struct('template_size', 36, 'hog_cell_size', 12);


%% Step 1. Load positive training crops and random negative examples
%YOU CODE 'get_positive_features' and 'get_random_negative_features'
[Xf, Yf] = get_features(train_path_pos, non_face_scn_path, feature_params_filter, -1, 100, {});

Xs = cell(0);
Ys = cell(0);

pos_infos = { {train_path_pos, {}},...
    {train_path_pos, {'rotate-left'}}, ...
    {train_path_pos, {'rotate-right'}},...
    {train_path_pos_sides, {}},...
    {train_path_pos_sides, {'flip'}},...
    {train_path_pos_sides, {'rotate-left'}},... 
    {train_path_pos_sides, {'flip','rotate-right'}} };
%pos_infos = { {train_path_pos, {}} };

imgs = cell(0);
for i = 1:length(pos_infos)
    imgs = [imgs ; get_positive_images(pos_infos{i}{1}, 36, 100, pos_infos{i}{2})];
end

for i = 1:length(pos_infos)
    imgs = get_positive_images(train_path_pos, feature_params, num, transforms);
    [Xs{i}, Ys{i}] = get_features(pos_infos{i}{1}, non_face_scn_path, feature_params, 10, 100, pos_infos{i}{2});
end
    
%% step 2. Train Classifiers

[Xf, Yf, filter] = train_classifier(Xf, Yf, train_path_pos, non_face_scn_path, feature_params_filter, 'regular', .00001);

classifiers = cell(0);

for i = 1:length(pos_infos)
    [Xs{i}, Ys{i}, classifiers{i}] = train_classifier(...
        Xs{i}, Ys{i}, pos_infos{i}{1}, non_face_scn_path, feature_params, 'regular', 0.00001);
end


%% step 3. Examine learned classifiers

fprintf('Filter classifier performance on train data:\n')
confidences = Xf'*filter.w + filter.b;
confidences = confidences';
label_vector = Yf;
[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy( confidences, label_vector );

% Visualize how well separated the positive and negative examples are at
% training time. Sometimes this can idenfity odd biases in your training
% data, especially if you're trying hard negative mining. This
% visualization won't be very meaningful with the placeholder starter code.
non_face_confs = confidences( label_vector < 0);
face_confs     = confidences( label_vector > 0);
figure(2); 
plot(sort(face_confs), 'g'); hold on
plot(sort(non_face_confs),'r'); 
plot([0 size(non_face_confs,1)], [0 0], 'b');
hold off;

% Visualize the learned detector. This would be a good thing to include in
% your writeup!
n_hog_cells = sqrt(length(filter.w) / 31); %specific to default HoG parameters
imhog = vl_hog('render', single(reshape(filter.w, [n_hog_cells n_hog_cells 31])), 'verbose') ;
figure(3); imagesc(imhog) ; colormap gray; set(3, 'Color', [.988, .988, .988]);
title('first pass filter to reduce computational costs');

pause(0.1) %let's ui rendering catch up
hog_template_image = frame2im(getframe(3));
% getframe() is unreliable. Depending on the rendering settings, it will
% grab foreground windows instead of the figure in question. It could also
% return a partial image.
imwrite(hog_template_image, ['visualizations/hog_template_filter.png'])

waitforbuttonpress;
close all;

for i = 1:length(pos_infos)
    % You don't need to modify anything in this section. The section first
    % evaluates _training_ error, which isn't ultimately what we care about,
    % but it is a good sanity check. Your training error should be very low.

    fprintf('Initial classifier performance on train data:\n')
    confidences = Xs{i}'*classifiers{i}.w + classifiers{i}.b;
    confidences = confidences';
    label_vector = Ys{i};
    [tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy( confidences, label_vector );

    % Visualize how well separated the positive and negative examples are at
    % training time. Sometimes this can idenfity odd biases in your training
    % data, especially if you're trying hard negative mining. This
    % visualization won't be very meaningful with the placeholder starter code.
    non_face_confs = confidences( label_vector < 0);
    face_confs     = confidences( label_vector > 0);
    figure(2); 
    plot(sort(face_confs), 'g'); hold on
    plot(sort(non_face_confs),'r'); 
    plot([0 size(non_face_confs,1)], [0 0], 'b');
    hold off;

    % Visualize the learned detector. This would be a good thing to include in
    % your writeup!
    n_hog_cells = sqrt(length(classifiers{i}.w) / 31); %specific to default HoG parameters
    imhog = vl_hog('render', single(reshape(classifiers{i}.w, [n_hog_cells n_hog_cells 31])), 'verbose') ;
    figure(3); imagesc(imhog) ; colormap gray; set(3, 'Color', [.988, .988, .988]);
    title(['path: ' pos_infos{i}{1} ', transform: ' pos_infos{i}{2}]);

    pause(0.1) %let's ui rendering catch up
    hog_template_image = frame2im(getframe(3));
    % getframe() is unreliable. Depending on the rendering settings, it will
    % grab foreground windows instead of the figure in question. It could also
    % return a partial image.
    imwrite(hog_template_image, ['visualizations/hog_template_' num2str(i) '.png'])

    waitforbuttonpress;
    close all;
end

%% Step 5. Run detector on test set.
% YOU CODE 'run_detector'. Make sure the outputs are properly structured!
% They will be interpreted in Step 6 to evaluate and visualize your
% results. See run_detector.m for more details.
[bboxes, confidences, image_ids] = run_detector(...
    test_scn_path, filter, feature_params_filter, classifiers, feature_params,...
    'step', 3, 'scales', [0.1, 0.3, 0.5, 0.8, 1.0, 1.2], 'threshold', 0.0, 'filter-threshold', 0.0);

% run_detector will have (at least) two parameters which can heavily
% influence performance -- how much to rescale each step of your multiscale
% detector, and the threshold for a detection. If your recall rate is low
% and your detector still has high precision at its highest recall point,
% you can improve your average precision by reducing the threshold for a
% positive detection.


%% Step 6. Evaluate and Visualize detections
% These functions require ground truth annotations, and thus can only be
% run on the CMU+MIT face test set. Use visualize_detectoins_by_image_no_gt
% for testing on extra images (it is commented out below).

% Don't modify anything in 'evaluate_detections'!
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path);

visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path)
% visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, test_scn_path)

% visualize_detections_by_confidence(bboxes, confidences, image_ids, test_scn_path, label_path);

% performance to aim for
% random (stater code) 0.001 AP
% single scale ~ 0.2 to 0.4 AP
% multiscale, 6 pixel step ~ 0.83 AP
% multiscale, 4 pixel step ~ 0.89 AP
% multiscale, 3 pixel step ~ 0.92 AP