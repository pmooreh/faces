function [ passed, score ] = test_down_tree( window, classifier_tree )

if isempty(classifier_tree)
    score = -1;
    passed = true;
else
    cell_size = classifier_tree{3};
    w = classifier_tree{4}{3};
    b = classifier_tree{4}{4};
    threshold = classifier_tree{4}{5};
    
    hog = vl_hog(window, cell_size);
    score = w'*reshape(hog, [], 1) + b;
    
    if score > threshold
        children = classifier_tree{1};
        for i = 1:length(children)
            [p, s] = test_down_tree(window, classifier_tree{1}{i});
            if p
                passed = p;
                score = max(s,score);
                return
            end
        end
        passed = false;
        score = 0;
    else
        passed = false;
        score = 0;
    end
end

end

