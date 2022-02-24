%分類する蕎麦の画像からDCNN特徴を抽出
n=0;list_test={};
IM = [];
net = alexnet;

LIST = {'soba_i'};
DIRO = 'imgdir/';

%画像の合体
for i=1:length(LIST)
    DIR = strcat(DIRO,LIST(i),'/');
    W = dir(DIR{:});

    for j=1:size(W)
        if(strfind(W(j).name,'.jpg'))
            fn = strcat(DIR{:},W(j).name);
            n=n+1;
            img = imread(fn);
             if(ndims(img) == 3)
                list_test={list_test{:} fn};
                reimg = imresize(img,net.Layers(1).InputSize(1:2));
                IM = cat(4,IM,reimg);
             end
        end
    end
end

%DCNN特徴を抽出
dcnnf_test = activations(net,IM,'fc7');
dcnnf_test = squeeze(dcnnf_test);
dcnnf_test = dcnnf_test/norm(dcnnf_test);
dcnnf_test = dcnnf_test';

%保存
save('list_soba2.mat','list_test');
save('dcnnf_soba2.mat','dcnnf_test');