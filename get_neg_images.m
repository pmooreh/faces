function [ imgs ] = get_neg_images(non_face_scn_path, template_size, num_samples)

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);
imgs = cell(num_samples, 1);

for i = 1:num_samples
    r = randi(num_images);
    rand_img = im2single(rgb2gray(imread([image_files(r).folder '/' image_files(r).name])));
    imgs{i} = imresize(random_box(rand_img), [template_size, template_size]);
end

end
