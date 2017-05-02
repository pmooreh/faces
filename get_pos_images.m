function [ imgs ] = get_pos_images(train_path_pos, template_size, num, transforms)

image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);
if num_images == 0
    image_files = dir( fullfile( train_path_pos, '*.pgm') ); %MIT Faces stored as .pgm
    num_images = length(image_files);
end
if num >= 0
    num = min(num_images, num);
else
    num = num_images;
end
imgs = cell(num, 1);
rands = randperm(num_images);

for i = 1:num
    r = rands(i);
    im = im2single(imread([image_files(r).folder '/' image_files(r).name]));
    im = imresize(im, [template_size, template_size]);
    for t = 1:length(transforms)
        switch transforms{t}
            case 'flip'
                im = flip(im,2);
            case 'rotate-left'
                im = imrotate(im, 20, 'crop');
            case 'rotate-right'
                im = imrotate(im, -20, 'crop');
        end
    end
    imgs{i} = im;
end

end

