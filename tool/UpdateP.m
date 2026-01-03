function [D_Mat] = UpdateP(Coef, Data, D_Mat)

Imat= eye(size(Coef,1)); %创建一个单位矩阵 Imat，维度与 Coef（G）的行数相同d*d
TempCoef = Coef;%Coef对应G
TempData = Data;%对应X
rho = 1;%rho 是 ADMM 的惩罚参数（增强变量一致性），初始值设为 1。
rate_rho = 1.2;%rate_rho 是 rho 的增长率，每次迭代后 rho 乘以 1.2 以加快收敛。
TempS = D_Mat;%对应投影矩阵F
TempT = zeros(size(TempS));%拉格朗日乘子 T（用于惩罚 F 和 假设的P 的不一致性）。
previousD = D_Mat;%存储上一轮的 F（D_Mat），用于计算误差。
Iter = 1;
ERROR=1;%误差初始设为 1，确保至少执行一次循环
while(ERROR>1e-6&&Iter<100)%误差 ERROR < 1e-6（即 D_Mat 收敛）代表10的-6次方
    %TempS = P  TempT = T TEMPd = F 下面是分别固定两个变量优化另一个变量
    TempD   = (rho*(TempS-TempT)'+TempData*TempCoef')/(rho*Imat+TempCoef*TempCoef');
    TempS   = normcol_lessequal(TempD'+TempT);
    TempT   = TempT+TempD'-TempS;
    rho     = rate_rho*rho;%每次迭代后，还将惩罚参数 rho 按比例增加（rho = rate_rho * rho），这有助于加快收敛过程。
    %计算当前迭代与上一次迭代 F 的变化误差
    ERROR = mean(mean((previousD- TempD').^2));
    previousD = TempD';
    %记录迭代次数
    Iter=Iter+1;
end
%返回迭代求得的F
D_Mat = TempD;
