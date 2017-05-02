% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, filter, feature_params_filter, classifiers, feature_params, varargin)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

suppress = true;
step_size = 6;
scales = [0.8, 1.0, 1.2];
threshold = 0.5;
filter_thresh = 0.5;
for i=1:2:length(varargin)
   switch varargin{i}
   case 'suppress'
     suppress = varargin{i+1};
   case 'scales'
     scales = varargin{i+1};
   case 'step'
     step_size = varargin{i+1};
   case 'threshold'
     threshold = varargin{i+1};
   case 'filter-threshold'
     filter_thresh = varargin{i+1};
   otherwise
     error(sprintf('%s is not a valid argument name',varargin{i}));
   end
end

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

for i = 1:length(test_scenes)
      
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    img_size = size(img);
    
    %You can delete all of this below.
    % Let's create 15 random detections per image
    cur_im_bboxes = [];
    cur_confidences = []; %confidences in the range [-2 2]
    cur_image_ids = [];
    
    % sliding window
    for s = 1:numel(scales)
        cur_scale_bboxes = [];
        scaled_im = imresize(img, scales(s));
        scaled_im_sz = size(scaled_im);
        num_steps = floor(scaled_im_sz / step_size);
        for x_step = 0:num_steps(2)-1
            for y_step = 0:num_steps(1)-1
                x_start = x_step * step_size;
                y_start = y_step * step_size;
                x_end = x_start + feature_params.template_size;
                y_end = y_start + feature_params.template_size;
                if (x_end > scaled_im_sz(2) || y_end > scaled_im_sz(1))
                    break
                end
                window = scaled_im(y_start+1:y_end, x_start+1:x_end);
                
                filter_hog = vl_hog(window, feature_params_filter.hog_cell_size);
                filter_score = filter.w'*reshape(filter_hog, [], 1) + filter.b;
                if filter_score > filter_thresh
                    best_score = -1.0;
                    for j = 1:length(classifiers)
                        hog = vl_hog(window, feature_params.hog_cell_size);
                        score = classifiers{j}.w'*reshape(hog, [], 1) + classifiers{j}.b;
                        best_score = max(score, best_score);
        %                   subplot(1,2,1);
        %                   imshow(window);
        %                   subplot(1,2,2);
        %                   imshow(vl_hog('render', hog));
        %                   title(num2str(score));
        %                   waitforbuttonpress;
        %                   close all;
                    end
                    if best_score > threshold
                        bbox = [x_start , y_start , x_end , y_end];
                        cur_scale_bboxes = [cur_scale_bboxes ; bbox];
                        cur_confidences = [cur_confidences; best_score];
                        cur_image_ids = [cur_image_ids;{test_scenes(i).name}];
                    end
                end
            end
        end
        cur_scale_bboxes = cur_scale_bboxes ./ scales(s);
        cur_im_bboxes = [cur_im_bboxes ; cur_scale_bboxes];
    end
    
    if numel(cur_confidences) > 0
        cur_im_bboxes = cur_im_bboxes + [1 1 0 0];
        x_out_of_bounds = cur_im_bboxes(:,3) > img_size(2);
        y_out_of_bounds = cur_im_bboxes(:,4) > img_size(1);
        cur_im_bboxes(x_out_of_bounds,3) = img_size(2);
        cur_im_bboxes(y_out_of_bounds,4) = img_size(1);
    
        %non_max_supr_bbox can actually get somewhat slow with thousands of
        %initial detections. You could pre-filter the detections by confidence,
        %e.g. a detection with confidence -1.1 will probably never be
        %meaningful. You probably _don't_ want to threshold at 0.0, though. You
        %can get higher recall with a lower threshold. You don't need to modify
        %anything in non_max_supr_bbox, but you can.
        if suppress
            [is_maximum] = non_max_supr_bbox(cur_im_bboxes, cur_confidences, size(img));

            cur_confidences = cur_confidences(is_maximum,:);
            cur_im_bboxes      = cur_im_bboxes(     is_maximum,:);
            cur_image_ids   = cur_image_ids(  is_maximum,:);
        end
    end
 
    bboxes      = [bboxes;      cur_im_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end




