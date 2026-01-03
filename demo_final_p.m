clear;
clc;
addpath('./data');
addpath('./tool');
addpath('./libsvm-new'); 
warning off;
rng('default');    % 恢复新生成器（如果之前有老接口操作）
rng('shuffle');
for testnum=[1]   %一共有12组实验，1:12
    switch testnum
        case 1
            name='be-CVE';
            load('berlin_feature-test219-end');
            load('berlin_label-test219-end');
            Xs=double(feature); clear feature



  
            Ys=double(label);   clear label
            load('CVE_feature-test351-end');
            load('CVE_label-test351-end');
        case 2
            name='be-IE';
            load('berlin_feature-test219-end');
            load('berlin_label-test219-end');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('IEMOCAP_feature-test2997-end');
            load('IEMOCAP_label-test2997-end');
        case 3
            name='be-TE';
            load('berlin_feature-test219-end');
            load('berlin_label-test219-end');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('TESS_feature-test1069-end');
            load('TESS_label-test1069-end');
        case 4
            name='CVE-be';
            load('CVE_feature-test351-end');
            load('CVE_label-test351-end');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('berlin_feature-test219-end');
            load('berlin_label-test219-end');
        case 5
            name='CVE-IE';
            load('CVE_feature-test351-end');
            load('CVE_label-test351-end');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('IEMOCAP_feature-test2997-end');
            load('IEMOCAP_label-test2997-end');
        case 6
            name='CVE-TE';
            load('CVE_feature-test351-end');
            load('CVE_label-test351-end');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('TESS_feature-test1069-end');
            load('TESS_label-test1069-end');
        case 7
            name='IE-be';
            load('IEMOCAP_feature-test2997-end');
            load('IEMOCAP_label-test2997-end');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('berlin_feature-test219-end');
            load('berlin_label-test219-end');
        case 8
            name='IE-CVE';
            load('IEMOCAP_feature-test2997-end');
            load('IEMOCAP_label-test2997-end');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('CVE_feature-test351-end');
            load('CVE_label-test351-end');
        case 9
            name='IE-TE';
            load('IEMOCAP_feature-test2997-end');
            load('IEMOCAP_label-test2997-end');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('TESS_feature-test1069-end');
            load('TESS_label-test1069-end');
        case 10
            name='TE-be';
            load('TESS_feature-test1069-end');
            load('TESS_label-test1069-end');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('berlin_feature-test219-end');
            load('berlin_label-test219-end');
        case 11
            name='TE-CVE';
            load('TESS_feature-test1069-end');
            load('TESS_label-test1069-end');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('CVE_feature-test351-end');
            load('CVE_label-test351-end');
        case 12
            name='TE-IE';
            load('TESS_feature-test1069-end');
            load('TESS_label-test1069-end');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('IEMOCAP_feature-test2997-end');
            load('IEMOCAP_label-test2997-end');
         case 13
            name='RML-EN';
            load('RML_5_1582');
            load('RML_5_label');
            Xs=double(feature); clear feature
            Ys=double(label);   clear label
            load('enterface_5_1582');
            load('enterface_5_label');
            
        otherwise
            break;
    end
    scale=0.7;
    nt=size(feature,1);
    Xtrain=double(feature(1:round(nt*scale),:));
    Ytrain=double(label(1:round(nt*scale),:));
    Xtest=double(feature(round(nt*scale)+1:end,:)); clear feature
    Yreal=double(label(round(nt*scale)+1:end,:));   clear label
    
    %% normalization and PCA
    Xs=normalization(Xs',1);
    Xs=Xs';
    Xtrain=normalization(Xtrain',1);
    Xtrain=Xtrain';
    Xtest=normalization(Xtest',1);
    Xtest=Xtest';
    X=[Xs;Xtrain;Xtest];
     
    % noise_level = 1e-3; % 如果波动不够，可以改为 1e-4；如果太大，改为 1e-6
    % X = X + noise_level * randn(size(X)); 
    
    [COEFF,SCORE, latent] = pca(X);
    SelectNum = cumsum(latent)./sum(latent);
    index = find(SelectNum >= 0.98);
    pca_dim = index(1);
    X=SCORE(:,1:pca_dim);
    
    Xs = X(1:size(Xs,1),:);
    Xtrain = X(size(Xs,1)+1:size(Xs,1)+size(Xtrain,1),:);
    Xtest = X(size(Xs,1)+size(Xtrain,1)+1:end,:);
    
    Y=[Ys;Ytrain];
    numClust=length(unique(Y));
    
    options=[];
    options.NeighborMode='KNN';
    options.WeightMode='Binary';
    options.k=5;
    options.max_iter = 50;
    options.dim=numClust;
    acc=0;
    acc_max=0;
    loop=0;
    cls_result=[];  %所有预测的标签
    obj=[]; %目标函数
    result=[];  %实验结果
    filename = mfilename;
    
    %% Experiments

    for alpha=0.01
        for lambda=100
            for beta=1
                % for rho=0
                    for gamma=100
                loop=loop+1;
                options.alpha=alpha;
                options.lambda=lambda;
                options.beta=beta;
                % options.rho=rho;
                options.gamma=gamma;
                
                
                [P,obj] = CDLSL_final_p(Xs,Xtrain,Ys,Ytrain,options,loop,obj);
                
                Zs=P'*Xs';   %c*n
                Zs = Zs*diag(sparse(1./sqrt(sum(Zs.^2))));
                Zt=P'*Xtest';
                Zt = Zt*diag(sparse(1./sqrt(sum(Zt.^2))));
                [~,cls] = max(Zt',[],2);
                acc = mean(Yreal == cls)*100;

                if acc>acc_max
                    acc_max=acc;
                end
                cls_result(:,loop)=cls;
                
                result(loop,1)=loop;
                result(loop,2)=acc;
                result(loop,3)=alpha;
                result(loop,4)=beta;
                result(loop,5)=lambda;
                % result(loop,6)=rho;
                result(loop,7)=gamma;

                disp([name,'      loop: ',num2str(loop),'      acc: ',num2str(acc),'      acc_max: ',num2str(acc_max)]);
                save(['./result/',name,'_',filename,'_obj1.mat'],'obj');  % 保存所有的目标函数值
                save(['./result/',name,'_',filename,'_cls_result1.mat'],'cls_result'); % 保存所有预测的标签结果
                save(['./result/',name,'_',filename,'_result1.mat'],'result');   %保存所有（准确率+超参结果）
                    end
                end
            % end
        end
    end
    
end
% 