clear;
clc;

addpath('./utils');

load('your image dir');
label = sta(img);

id = 'non_sv';
dname = 'CAVE';

save_dir = 'where the created images are saved';

ful_n = [dname '_' id];
if ~exist([save_dir '/' ful_n],'dir')
    mkdir([save_dir '/' ful_n])
end

switch id
    case 'iid(30)'
        input = label + (30/255)*randn(size(label)); 
        save([save_dir '/' ful_n '/' ful_n '.mat'], 'input','label')
    case 'non'
        [input, sigma] = add_noniid(label);
        save([save_dir '/' ful_n '/' ful_n '.mat'], 'input','label','sigma')
    case 'non_sv'
        [input, sigmap] = add_gaunon(label);
        save([save_dir '/' ful_n '/' ful_n '.mat'], 'input','label','sigmap')
    case 'stripe'
        [input, band, sn] = add_stripe(label,10,0.05,0.2);
        save([save_dir '/' ful_n '/' ful_n '.mat'], 'input','label','band','sn')
    case 'impulse'
        [input, band, ratio] = add_impulse(label, 10);
        save([save_dir '/' ful_n '/' ful_n '.mat'], 'input','label','ratio','band')
    case 'deadline'
        [input, band, dn] = add_deadline(label, 10, 0.05,0.2);
        save([save_dir '/' ful_n '/' ful_n '.mat'], 'input','label','band','dn')
    case 'mix'
        input = add_mixture(label, 10);
        save([save_dir '/' ful_n '/' ful_n '.mat'], 'input','label')
    otherwise
        disp('unrecognized type!')
end


