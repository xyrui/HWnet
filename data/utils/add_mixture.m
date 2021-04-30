function input = add_mixture(label, bn)
[~,W,B] = size(label);
tB = rand(1,B);
B1 = find(tB<0.5);
B2 = find(tB>=0.5);

input(:,:,B1) = add_noniid(label(:,:,B1));
input(:,:,B2) = add_gaunon(label(:,:,B2));

% add stripe
band = randperm(B, bn);               
stripnum = randi([ceil(0.05 * W), ceil(0.2 * W)], length(band), 1); 
for i=1:length(band)
    loc = randperm(W, stripnum(i));
    stripe = rand(1,length(loc))*0.5-0.25;
    input(:,loc,band(i)) = input(:,loc,band(i)) - stripe;
end

% add impluse
band = randperm(B, bn);
ratio = rand(bn,1)*0.4+0.1;
 for i=1:bn
     input(:,:,band(i)) = imnoise(input(:,:,band(i)),'salt & pepper',ratio(i));
 end
 
% add deadline
band = randperm(B, bn);
deadlinenum = randi([ceil(0.05 * W), ceil(0.2 * W)], bn, 1);
for i=1:bn
    loc = randperm(W, deadlinenum(i));
    input(:,loc,band(i)) = 0;
end

end