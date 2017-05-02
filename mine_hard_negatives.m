function [ hard_negs ] = mine_hard_negatives( non_face_scn_path, w, b, feature_params, num )

[res, scores] = mine_negatives(non_face_scn_path, w, b, feature_params, num);
hard_negs = res(scores > 0,:);

end

