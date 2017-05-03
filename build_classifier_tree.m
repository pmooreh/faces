function [ classifier_tree ] = build_classifier_tree( feature_tree, threshold )

if isempty(feature_tree)
    classifier_tree = {};
else
    X = [feature_tree{4}' , feature_tree{5}'];
    Y = [ones(1, size(feature_tree{4}, 1)), -1*ones(1, size(feature_tree{5}, 1))];
    [w, b] = vl_svmtrain(X, Y, 0.00001);
    children = cell(1, length(feature_tree{1}));
    for i = 1:length(children)
        children{i} = build_classifier_tree(feature_tree{1}{i}, threshold);
    end
    classifier_tree = {children, 'bad-code', feature_tree{3}, {X, Y, w, b, threshold}};
end

end

