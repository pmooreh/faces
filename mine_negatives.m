function [ res, scores ] = mine_negatives( non_face_scn_path, w, b, feature_params, num )

res = zeros(num,...
    (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
scores = zeros(num, 1);

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);

for i = 1:num
    r = randi(num_images);
    rand_img = im2single(rgb2gray(imread([image_files(r).folder '/' image_files(r).name])));
    rand_patch = imresize(random_box(rand_img),...
        [feature_params.template_size, feature_params.template_size]);
    hog = vl_hog(rand_patch, feature_params.hog_cell_size);
    scores(i) = w'*reshape(hog, [], 1) + b;
    res(i,:) = reshape(hog, 1, []);
end

end

