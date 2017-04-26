function [ box ] = random_box(img)

rand_length = randi(min(size(img, 1), size(img, 2))) - 1;
coord_dim = size(img) - rand_length;
rand_ind = randi(prod(coord_dim));
[r, c] = ind2sub(coord_dim, rand_ind);
box = img(r:r+rand_length, c:c+rand_length);

end

