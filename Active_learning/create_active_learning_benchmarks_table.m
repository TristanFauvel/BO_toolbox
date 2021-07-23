clear all
add_gp_module;
fName = {'ackley'; 'beale'; 'boha1'; 'camel3'; 'camel6'; 'colville'; 'crossit'; ...
    'dixonpr'; 'drop'; 'egg'; 'forretal08'; 'goldpr'; 'griewank'; 'grlee12';  ...
    'hart3'; 'hart4'; 'hart6'; 'holder'; 'langer';'levy';'levy13';'perm0db';'permdb';...
    'powell'; 'rosen'; 'rothyp'; 'schaffer4'; 'schwef'; 'shekel'; 'shubert'; ...
    'spheref'; 'sumsqu'; 'trid'; 'Ursem_waves'};

N = numel(fName);
% Matern32 = NaN(N,2);
% Matern52 = NaN(N,2);

for i = 1:N
    fun = str2func(fName{i});
    fun = fun();
    D(i,1) = fun.D;
    Name{i,1} = fun.name;
    ARD{i,1} = [];
    Matern32{i,1} = [];
    Matern52{i,1} = [];    
end
benchmarks_table = table(Name, fName, D, ARD, Matern32, Matern52);

table2latex(benchmarks_table, [pathname, '/benchmarks.tex'])

fName = categorical(fName);
Name = categorical(Name);
benchmarks_table = table(Name, fName, D, ARD, Matern32, Matern52);

Kernel_name = categorical(benchmarks_table.Kernel_name)
benchmarks_table.Kernel_name  = Kernel_name ;

Kernel= categorical(benchmarks_table.Kernel)
benchmarks_table.Kernel  = Kernel ;

save([pathname, '/Benchmarks/benchmarks_table.mat'],'benchmarks_table')


load([pathname, '/Benchmarks/benchmarks_table.mat'],'benchmarks_table')

% benchmarks_table(benchmarks_table.Name == 'Powell',:).ARD = {zeros(5,1)};