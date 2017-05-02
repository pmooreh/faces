function [ res, scores ] = mine_positives( face_scn_path, w, b, feature_params, num )

image_files = dir( fullfile( face_scn_path, '*.jpg' ));
num_images = length(image_files);
num = min(num, num_images);

res = zeros(num,...
    (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);
scores = zeros(num, 1);

for i = 1:num
    r = randi(num_images);
    rand_img = im2single(imread([image_files(r).folder '/' image_files(r).name]));
    hog = vl_hog(rand_img, feature_params.hog_cell_size);
    scores(i) = w'*reshape(hog, [], 1) + b;
    res(i,:) = reshape(hog, 1, []);
end

end

