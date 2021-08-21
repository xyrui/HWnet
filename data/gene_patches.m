% Readme: Create spatially over-lapped patches

clear;

D = dir('./matfile/*.mat');

k = 1; % counting
h_pn = 8; % patch_num (vertical)
w_pn = 8; % patch_num (horizontal)
b_pn = 1;

p_s = 96;  % patch size (spatial)
p_b = 20;  % patch size (spetral)

for i = 1:20
    load(['./matfile/' D(i).name]) % File where I save CAVE dataset
    rad = sta(A);   % normalization band by band
    [MM,NN,BB] = size(rad);
    
    t_h = ceil((h_pn*p_s- MM)/(h_pn-1)); 
    t_w = ceil((w_pn*p_s - NN)/(w_pn-1));
%     t_b = ceil((b_pn*p_b - BB)/(b_pn-1)); 
    
    for H = 1:h_pn
        for W = 1:w_pn
            for B = 1:b_pn
            
                %if b_pn equals 1£¬then randomly select adjacent bands
                t_b = randi(BB-p_b);
           
            u = 1;
            patch_ = rad(1+(H-1)*(p_s-t_h):H*p_s-t_h*(H-1), 1+(W-1)*(p_s-t_w):W*p_s-t_w*(W-1), t_b:t_b+p_b-1);  %1+(B-1)*(p_b-t_b):B*p_b-t_b*(B-1)
            patch = patch_;

            save(['./HWLRMF_patches/p_' num2str((k-1)*8+u) '.mat'],'patch')
            u = u+1;
            patch = patch_(:,end:-1:1,:); % spatial flip

            save(['./HWLRMF_patches/p_' num2str((k-1)*8+u) '.mat'],'patch')
            u = u+1;
            for rota = 1:3  % spatial rotation
                patch = rot90(patch_,rota);

                save(['./HWLRMF_patches/p_' num2str((k-1)*8+u) '.mat'],'patch')
                u = u+1;
                patch = rot90(patch_(:,end:-1:1,:),rota);

                save(['./HWLRMF_patches/p_' num2str((k-1)*8+u) '.mat'],'patch')
                u = u+1;         
            end
            disp([num2str(i) ' [' num2str(k*8) ']/[110000]'])
            k = k+1;   
            if k*8>10000
                break
            end
            end
        end
    end
    clear A      
end

%% temp(for DSSnet)
% h_pn = 11; 
% w_pn = 11;  
% b_pn = 3;
% 
% p_s = 48;  
% p_b = 10;  