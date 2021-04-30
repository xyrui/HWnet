function [im, band, stripnum] = add_stripe(im, bn, min_amount, max_amount)  
[~, N, B] = size(im);
band = randperm(B, bn);

sigma = randi(60, B,1) + 10;       
s = reshape(sigma, 1, 1, length(sigma));
im = im + s/255 .* randn(size(im));
                
stripnum = randi([ceil(min_amount * N), ceil(max_amount * N)], length(band), 1); 
% disp(stripnum);
for i=1:length(band)
    loc = randperm(N, stripnum(i));
    stripe = rand(1,length(loc))*0.5-0.25;
    im(:,loc,band(i)) = im(:,loc,band(i)) - stripe;
end
end
