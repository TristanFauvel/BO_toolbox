function [best_x, best_val] = optimize_AF(fun, lb, ub, ncandidates, init_guess, options, varargin)
% To optimize acquisition functions
opts = namevaluepairtostruct(struct( ...
    'dims', [], ...
    'objective', 'max' ...
    ), varargin);

UNPACK_STRUCT(opts, false)

%% Multistart seach with minFunc

s.verbose = 0;

DEFAULT('ncandidates',10);
DEFAULT('options', s);
best_x= [];

x0 = lb;
f = @(x) subfun(x, fun,dims, x0, objective);
if ~isempty(dims)
    lb = lb(dims);
    ub = ub(dims);
end
 % Batch initialization for multi-start optimization : select starting points using the method by
% Balandat et al 2020 (appendix F).
D = size(x0,1);
d = size(lb,1);

if isempty(dims)
dims = 1:D;
end

p = haltonset(d,'Skip',1e3,'Leap',1e2);
p = scramble(p,'RR2');
q = qrandstream(p);
n0 = 500;
sampSize = 1;
X = x0.*ones(1, n0);
v = zeros(1,n0);
for i = 1:n0
    X(dims,i) = qrand(q,sampSize);
    v(i) = fun(X(:,i));
end

temperature =  1;
p  = exp(temperature*zscore(v));
p = p/sum(p);
starting_points = X(dims,randsample(n0, ncandidates, true, p));

%starting_points = rand_interval(lb, ub, 'nsamples', ncandidates);
if ~isempty(init_guess)
    starting_points(:,1)= init_guess;
end

if strcmp(objective, 'max')
    best_val=-inf;
elseif strcmp(objective, 'min')
    best_val=inf;
end

x = [];
for k = 1:ncandidates
    try
        x = minConf_TMP(@(x)f(x), starting_points(:,k), lb(:), ub(:), options);
        if any(isnan(x(:)))
            error('x is NaN')
        end
        val = f(x);

        if strcmp(objective, 'max')
            val = -val;
        end

        if (strcmp(objective, 'min') && val < best_val) || (strcmp(objective, 'max') && best_val < val)
            best_x = x;
            best_val = val;
        end
    catch e %e is an MException struct
        fprintf(1,e.message);
    end
end

if isempty(x)
    error('Optimization failed')
end

x0(dims) = best_x;
best_x = x0;
end

function [f, df] = subfun(x, fun,dims, x0, objective)

if ~isempty(dims)
    x0(dims) = x;
    [f, df] = fun(x0);
    df = df(dims);
else
    [f, df] = fun(x);
    df = df(:);
end

if strcmp(objective, 'max')
    f = -f;
    df = -df;
end
end
