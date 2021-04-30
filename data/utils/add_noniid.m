function [im,s] = add_noniid(im)
[H,W,B] = size(im);
sigma = randi(60, B,1) + 10;       
s = reshape(sigma, 1, 1, length(sigma))/255;
im = im + s.* randn(size(im));
end