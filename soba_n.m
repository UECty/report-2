%学習のポジティブ画像とする蕎麦の画像からDCNN特徴を抽出
n=0;list={};
IM = [];
net = alexnet;

LIST = {'soba_r'};
DIRO = 'imgdir/';
%画像を合体
for i=1:length(LIST)
    DIR = strcat(DIRO,LIST(i),'/');
    W = dir(DIR{:});

    for j=1:size(W)
        if(strfind(W(j).name,'.jpg'))
            fn = strcat(DIR{:},W(j).name);
            n=n+1;
            
            list={list{:} fn};

            img = imread(fn);
            reimg = imresize(img,net.Layers(1).InputSize(1:2));
            IM = cat(4,IM,reimg);
        end
    end
end

%DCNN特徴を抽出
dcnnf = activations(net,IM,'fc7');
dcnnf = squeeze(dcnnf);
dcnnf = dcnnf/norm(dcnnf);
dcnnf = dcnnf';

%保存
save('list_soba1.mat','list');
save('dcnnf_soba1.mat','dcnnf');