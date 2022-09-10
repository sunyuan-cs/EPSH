clear all
close all



% load dataset
fprintf('Loading Data...\n');
load('mnist.mat')
% fix seed
seed = 0;
rng('default');
rng(seed);
param.seed = seed;

traingnd=traingnd+ones(length(traingnd),1);
testgnd=testgnd+ones(length(testgnd),1);


exp_data.traingnd=traingnd;
exp_data.testgnd=testgnd;
cateTrainTest = bsxfun(@eq, traingnd, testgnd');
exp_data.WTT=cateTrainTest';
exp_data.traindata = double(traindata);
exp_data.testdata = double(testdata);


total_res=[];
top_K=5000;

nbits_set =[4,6,8,10];
%parameters for mnist
candidate_alpha= [1e-4];%[1e-4,1e-3,1e-2,1e-1,1];%
candidate_beta=[1e3];%[1,1e1,1e2,1e3,1e4];%

for ii=1:length(nbits_set)
    for c_alpha = 1: length(candidate_alpha)
        for c_beta = 1: length(candidate_beta)
            paras.alpha = candidate_alpha(c_alpha);
            paras.beta = candidate_beta(c_beta)
            nbits=nbits_set(ii);
            [MAP,precision,recall,Precision_top,time] =train_EPH(exp_data, nbits, paras.alpha,paras.beta,top_K);
            fprintf('MAP result of EPSH: %d...   \n', MAP );
            fprintf('Precision_top result of EPSH: %d...   \n', Precision_top );
            total_res = [total_res; MAP,Precision_top,paras.alpha, paras.beta, nbits,time];
        end
    end
    
end
save('EPH_mnist.mat','total_res');



