function [im, band, deadlinenum] = add_deadline(im, bn, min_amount, max_amount)
[~, N, B] = size(im);
band = randperm(B, bn);

sigma = randi(60, B,1) + 10;       
s = reshape(sigma, 1, 1, length(sigma));
im = im + s/255 .* randn(size(im));

deadlinenum = randi([ceil(min_amount * N), ceil(max_amount * N)], bn, 1);

for i=1:bn
    loc = randperm(N, deadlinenum(i));
    im(:,loc,band(i)) = 0;
end
end