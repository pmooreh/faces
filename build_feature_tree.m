function [ feat_tree ] = build_feature_tree( pos_imgs, neg_imgs, descrs )

if isempty(descrs)
    feat_tree = cell(0);
else
    feature_params = struct('template_size', 36, 'hog_cell_size', descrs{1}{1});
    pos_feats = get_positive_features(feature_params, pos_imgs);
    neg_feats = get_positive_features(feature_params, neg_imgs);
    
    idx = kmeans(pos_feats, descrs{1}{2});
    
    children = cell(1, descrs{1}{2});
    
    for i = 1:length(children)
        imgs = pos_imgs(idx == i);
        children{i} = build_feature_tree(imgs, neg_imgs, descrs(2:end));
    end
    
    feat_tree = {children, 'bad-coding', descrs{1}{1}, pos_feats, neg_feats};
end

end

