function [P,obj] = DGCWSL(Xs,Xt,Ys,Yt,options,loop,obj)
% 主项为图

% ------------ Initialization  ---------- %
% rand('seed',6666);
rng('default');    % 恢复新生成器（如果之前有老接口操作）
rng('shuffle');
X=[Xs;Xt];
X=X';
Xs=Xs';
Xt=Xt';
eps = 1e-8;
[d,n] = size(X);
[~,ns] = size(Xs);
[~,nt] = size(Xt);
dim=options.dim;
v=ones(1,d)./d;
% ------------ Initialize W ----------------- %
options.ReducedDim = dim;
[P1,~] = PCA1(X', options);
W = P1;
% ---------- Initialize other parameters ----------------- %
Z = ones(ns,nt);
vv=sqrt(sum(Z.*Z,2)+eps);
Dz=diag(1./(vv));

W1 = constructW(Xs',options);
   W1=full(W1);
   DD1 = diag(sum(W1));
    Lw1 = DD1 - W1;
   Lw1 = Lw1 / norm(Lw1,'fro');

% R=constructRRR(X',5);
P = inv(diag(v))*W;
% ------------ Initialize Ytrain_pseudo ------------- %
% model=svmtrain(Ys,Xs,'-s 0 -t 0 -c 1 -g 1 ');
% [Ytrain_pseudo, ~, ~] = svmpredict(Yt,Xt,model);
% YY=[Ys;Ytrain_pseudo];
% Y = Pre_label(YY);
Ys=Pre_label(Ys);


%% iteration
for iter = 1:options.max_iter
    


    ed = L2_distance_1(W'*X, W'*X);
        parfor j = 1:n
            sd = (-options.alpha*(ed(j,:)))/(2);
            S(j,:) = EProjSimplex_new(sd);
        end
    
    % S = S + 0.01 * rand(n,n);
    % S(1:n+1:end) = 0;
    % S = S ./ sum(S,2);

    temp2_S = (S+S')*0.5;
    Sum_S = sum(temp2_S);
    LS = diag(Sum_S)-temp2_S;
    % ----------------update W ----------------- %
    Dv = diag(1./(v.^2+ eps));
    V1 = Xt-Xs*Z;
    
    % [W,~]=eigs(options.beta*X*LS*X'+V1*(V1)'+options.lambda*Dv,X*X',dim,'SM');
    % W=(options.beta*X*LS*X'+options.gamma*V1*(V1)'+options.lambda*Dv+Xs*Xs')\(Xs*Ys);
    W=(X*LS*X'+options.gamma*V1*(V1)'+options.lambda*Dv+options.beta*Xs*Xs')\(options.beta*Xs*Ys);
    % -----------------update v ------------------- %
    dd=sqrt(sum(W.^2,2))+ eps;
    v=dd./sum(dd);
    v=v';

    % ------------- update Ytrain_pseudo ------------- %
    Z = (options.gamma*Lw1+options.gamma*Xs'*W*W'*Xs)\(options.gamma*Xs'*W*W'*Xt);
    % Z = (Xs'*W*W'*Xs)\(Xs'*W*W'*Xt);
    vv  = sqrt(sum(Z.*Z,2)+eps);
    Dz  = diag(1./(vv));
    % Zt = P'*Xt';
    % Zt = Zt*diag(sparse(1./sqrt(sum(Zt.^2))));
    % [~,Ytrain_pseudo] = max(Zt',[],2);  %更新伪标签
    % Y = [Ys;Ytrain_pseudo];
    % Y = Pre_label(Y);
    
    % -----------------update B ------------------- %
    P = inv(diag(v))*W;


    % -------------- obj --------------- %
    Item1=options.gamma*norm(W'*Xt-W'*Xs*Z,'fro')^2;
    Item2=options.alpha*norm(S,'fro')^2;    %options.alpha*norm(Z-(Y+B⊙M),'fro')^2
    Item3=options.lambda*norm(P,'fro')^2;
    Item4=trace(W'*X*LS*X'*W);
    Item5=options.gamma*trace(Z'*Lw1*Z);
    Item6=options.beta*norm(W'*Xs-Ys','fro')^2;
          
    obj(loop,iter) = (Item1+Item2+Item3+Item4+Item5+Item6); 
    if iter >5 && abs(obj(loop,iter)-obj(loop,iter-1))<1e-3
        break;
    end
    % disp(['Iter: ',num2str(iter),'obj:  ',num2str(obj(iter),'%.10f')]);
end

    disp(['Iter: ',num2str(iter)]);
end
