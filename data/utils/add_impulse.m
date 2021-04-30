function [im, band, ratio] = add_impulse(im, bn)
[~, ~, B] = size(im);
band = randperm(B, bn);

sigma = randi(60, B,1) + 10;       
s = reshape(sigma, 1, 1, length(sigma));
im = im + s/255 .* randn(size(im));

ratio = rand(bn,1)*0.4+0.1;

 for i=1:bn
     im(:,:,band(i)) = imnoise(im(:,:,band(i)),'salt & pepper',ratio(i));
 end
 
end