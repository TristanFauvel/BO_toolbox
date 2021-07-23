clear all
add_gp_module;
fName = {'japan_matern52_short','japan_matern32_short','japan_sexp_short', ...
    'japan_matern52_long', 'japan_matern32_long', 'japan_sexp_long'};

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
active_learning_benchmarks_table = table(Name, fName, D, ARD, Matern32, Matern52);

table2latex(active_learning_benchmarks_table, [pathname, '/active_learning_benchmarks.tex'])

fName = categorical(fName);
Name = categorical(Name);
active_learning_benchmarks_table = table(Name, fName, D, ARD, Matern32, Matern52);

Kernel_name = categorical(active_learning_benchmarks_table.Kernel_name)
active_learning_benchmarks_table.Kernel_name  = Kernel_name ;

Kernel= categorical(active_learning_benchmarks_table.Kernel)
active_learning_benchmarks_table.Kernel  = Kernel ;

save([pathname, '/active_learning_benchmarks/active_learning_benchmarks_table.mat'],'active_learning_benchmarks_table')


load([pathname, '/active_learning_benchmarks/active_learning_benchmarks_table.mat'],'active_learning_benchmarks_table')

% active_learning_benchmarks_table(active_learning_benchmarks_table.Name == 'Powell',:).ARD = {zeros(5,1)};