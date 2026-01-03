function [R] = constructRRR(fea, multiorder)

%% LPP 参数设置（用于构建图的局部保持投影参数）
options = [];
options.NeighborMode = 'KNN';       % 使用 K-近邻方式构建图
options.k = 5;                      % 每个样本的邻居数量 k
options.WeightMode = 'HeatKernel';  % 使用热核（高斯核）计算边权重
options.t = 1;                      % 热核参数 t（带宽）

% 输入：
%   fea        - 大小为 n × d 的单视图特征矩阵
%   multiorder - 随机游走的最大步数 Q，用于累加 A^1 + A^2 + ... + A^Q

[n, ~] = size(fea); % 样本数量 n

% 1) 构建一阶相似矩阵 A（KNN + 热核）
A = constructW(fea, options);

% 2) 初始化累加矩阵 sum，用于存放 A + A^2 + ... + A^Q
sum = zeros(n);

% 3) 随机游走：累加各阶邻接矩阵 A^num2
for num2 = 1 : multiorder
    sum = sum + A ^ num2; % 计算矩阵幂并累加
end

% 4) 去除自环：将对角元素置零，避免自连接影响
sum = sum - diag(diag(sum));

% 5) 归一化到 [0,1] 区间，增强数值稳定性
R = mapminmax(sum, 0, 1);

end
