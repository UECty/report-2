%ネガティブ画像とするランダム画像からDCNN特徴を抽出
n=0;list_r2={};
IM = [];
net = alexnet;

LIST = {'bgimg'};
DIRO = 'imgdir/';

%画像を合体
for i=1:length(LIST)
    DIR = strcat(DIRO,LIST(i),'/');
    W = dir(DIR{:});

    for j=301:size(W)
        if(strfind(W(j).name,'.jpg'))
            fn = strcat(DIR{:},W(j).name);
            n=n+1;
            
            list_r2={list_r2{:} fn};

            img = imread(fn);
            reimg = imresize(img,net.Layers(1).InputSize(1:2));
            IM = cat(4,IM,reimg);
        end
    end
end

%DCNN特徴を抽出
dcnnf_r2 = activations(net,IM,'fc7');
dcnnf_r2 = squeeze(dcnnf_r2);
dcnnf_r2 = dcnnf_r2/norm(dcnnf_r2);
dcnnf_r2 = dcnnf_r2';

%保存
save('list_rand2.mat','list_r2');
save('dcnnf_rand2.mat','dcnnf_r2');