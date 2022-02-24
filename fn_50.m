load('dcnnf_soba1.mat');
load('dcnnf_soba2.mat');
load('dcnnf_rand.mat');
load('dcnnf_rand2.mat');
load('list_soba2.mat');

OUTDIR='imgdir/res_50'

list = list_test(:,1:300);%分類画像を300枚にする
data_pos = dcnnf(1:50,:);%ポジティブ画像を300枚にする
data_neg = [dcnnf_r;dcnnf_r2];
data_neg = data_neg(1:500,:);%ネガティブ画像を500枚にする

train = [data_pos; data_neg];
eval = dcnnf_test(1:300,:);
train_label = [ones(50,1); ones(500,1)*(-1)];

%線形SVMによる分類
model = fitcsvm(train,train_label,'KernelFunction','linear');
[predicted_label,scores] = predict(model,eval);

%ソート
[sorted_score,sorted_idx] = sort(scores(:,2),'descend');

%上位100画像を保存
for i=1:100
    copyfile(list{1,sorted_idx(i)},OUTDIR);
    fprintf('[%d] %s %f\n',i,list{1,sorted_idx(i)}, sorted_score(i));
end