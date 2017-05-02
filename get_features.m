function [ X, Y ] = get_features( train_path_pos, non_face_scn_path, feature_params, n_pos, n_neg, pos_trans )

features_pos = get_positive_features( train_path_pos, feature_params, n_pos, pos_trans);
features_neg = get_random_negative_features( non_face_scn_path, feature_params, n_neg);

tic
idx = kmeans(features_pos, 2);
toc

X = [features_pos' , features_neg' ];
Y = [ones(1, size(features_pos, 1)), -1*ones(1, size(features_neg, 1))];

end

