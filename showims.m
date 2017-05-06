function [  ] = showims( ims )

for i = 0:5
    subplot(1,6,1+i);
    imshow(ims{i*100 + 1});
end

end

