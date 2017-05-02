function [ X, Y, c] = train_classifier( X, Y, train_path_pos, non_face_scn_path, feature_params, how, lambda )

switch how
    case 'regular'
        [w b] = vl_svmtrain(X, Y, lambda);
    case 'boundary-expansion'
        [w b] = vl_svmtrain(X, Y, lambda);
        
        for i = 1:20
            mine = 1000;
            [res_neg, scores_neg] = mine_negatives(non_face_scn_path, w, b, feature_params, mine);
            [res_pos, scores_pos] = mine_positives(train_path_pos, w, b, feature_params, mine);
            
            [~, IN] = sort(scores_neg, 'descend');
            [~, IP] = sort(scores_pos, 'descend');
            
            res_neg = res_neg(IN,:);
            res_pos = res_pos(IP,:);
            
            keep = 100;
            X = [res_pos(1:keep,:)' , res_neg(1:keep,:)'];
            Y = [ones(1,keep) , -1*ones(1,keep)];
            [w b] = vl_svmtrain(X, Y, lambda);
        end
    case 'mine-hard-negs'
        [w b] = vl_svmtrain(X, Y, lambda);
        
        hard_negs = mine_hard_negatives(non_face_scn_path, w, b, feature_params, 100000);
        X = [X, hard_negs'];
        Y = [Y, -1*ones(1, size(hard_negs, 1))];
        [w b] = vl_svmtrain(X, Y, lambda);
    otherwise
        error([how 'is not a valid training style']);
end

c = struct('w', w, 'b', b);

end

