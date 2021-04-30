function [input, psig] = add_gaunon(label)
[H,W,B] = size(label);

% gaussian_kernel1
smax = 128*(H/256);  % basic spatial size 256,smax=128,smin=32
smin = 32*(H/256);
scale = rand(1,1,B)*(smax-smin)+smin;
psig = gaussian_kernel1(H,W,B,scale);

% % gaussian_kernel2
% smax = 100;
% smin = 20;
% scale = rand(1,1,B)*(smax - smin) + smin;
% psig = gaussian_kernel2(H,W,B,scale);

pmax = 70/255;
pmin = 10/255;
psig = (psig-min(psig(:)))/(max(psig(:))-min(psig(:)))*(pmax-pmin)+pmin;
input = label + randn(H,W,B).*psig;
end

function psig = gaussian_kernel1(H,W,B,scale)
centerSpa1 = randi(H-10,1,1,B)+5;
centerSpa2 = randi(W-10,1,1,B)+5;
[XX,YY] = meshgrid(1:W,1:H);
psig = exp((-(repmat(XX,[1,1,B])-centerSpa1).^2-(repmat(YY,[1,1,B])-centerSpa2).^2)./(2*scale.^2));
end

function psig = gaussian_kernel2(H,W,B,scale)
[XX,YY] = meshgrid(1:W,1:H);
psig = sin(repmat(XX,[1,1,B])./scale);
end
