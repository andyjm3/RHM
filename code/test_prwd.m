% test on projection robust wasserstein distance
% f = 1/n^2 sum_ij (pi_ij || U' x_i - U' x_j ||^2 + reg * pi_ij log(pi_ij)
% In our notation, min_x max_y where x is pi and y is U.

function test_prwd()
    clc; clear;
    
    rng(99);
    
    % data on hypercube (modified from
    % https://github.com/fanchenyou/PRW/blob/master/exp1_hypercube.py)
    n = 100; % samples 
    d = 30;
    k = 5;
    dim = 2;
    reg = 0.2; % regularization
    
    % histograms
    a = (1. / n) * ones(n,1);
    b = (1. / n) * ones(n,1);
        
    % First measure : uniform on the hypercube between [-1, 1]
    X = -1 + 2*rand(n,d);
    % Second measure : fragmentation
    Y = -1 + 2*rand(n,d);
    temp = cat(1, ones(dim, 1), zeros(d - dim, 1));
    Y = Y + 2 * sign(Y) * temp;
    %}

    %%
    Mu = stiefelfactory(d, k);
    Mpi = multinomialdoublystochasticgeneralfactory(n,n,a,b);

    problem.Mx = Mu;
    problem.My = Mpi;
    problem.M = productmanifold(struct('x', Mpi , 'y', Mu));
    problem.cost = @cost;
    problem.egrad = @egrad;
    problem.ehess = @ehess;
    %problem.hess = approxhessianFD(problem);

    function f = cost(xy)
        pi = xy.x; % pi
        U = xy.y; % U
        UUT = U*U';
    
        Dmat = diag(X*UUT*X') * ones(1,n) + ones(n,1) * diag(Y*UUT*Y')' ...
            - 2*X*UUT*Y';
        
        f = sum(pi(:) .* Dmat(:))/(n^2) + reg * sum(pi(:) .* log(pi(:)) - pi(:));
    end
    
    
    function g = egrad(xy)
        pi = xy.x; % pi
        U = xy.y; % U
        UUT = U*U';

        Dmat = diag(X*UUT*X') * ones(1,n) + ones(n,1) * diag(Y*UUT*Y')' ...
            - 2*X*UUT*Y';
        tempg = X'*pi*Y;
        sumc = sum(pi, 2);
        sumr = sum(pi, 1);
        Vpi = X'*diag(sumc)*X + Y'*diag(sumr)*Y -tempg -tempg'; % second order displacement
    
        g.x = Dmat./(n^2) + reg * log(pi);
        g.y = (2/(n^2)) .* (Vpi*U);
    end

    function h = ehess(xy, xydot)
        pi = xy.x; % pi
        U = xy.y; % U
        pidot = xydot.x;
        Udot = xydot.y;

        tempg = X'*pi*Y;
        sumc = sum(pi, 2);
        sumr = sum(pi, 1);
        Vpi = X'*diag(sumc)*X + Y'*diag(sumr)*Y -tempg -tempg'; % second order displacement
        
        temph = X'*pidot*Y;
        sumc = sum(pidot, 2);
        sumr = sum(pidot, 1);
        DVpi = X'*diag(sumc)*X + Y'*diag(sumr)*Y -temph - temph';
        
        DUUt = Udot * U' + U * Udot';
        DDmat = diag(X*DUUt*X') * ones(1,n) + ones(n,1) * diag(Y*DUUt*Y')' ...
            - 2*X*DUUt*Y';

        h.x = reg * (pidot ./ pi) + DDmat./(n^2);
        h.y = (2/(n^2)) .* (Vpi * Udot + DVpi*U); % ok

    end
    %checkgradient(problem); %ok
    %checkhessian(problem); %ok up to numerical errors
    %keyboard;

    xy0 = problem.M.rand();
    maxiter = 300;
    
    %%
    % RHM-fixedstep
    optionsRHMsdf.stepsize = 42;
    optionsRHMsdf.gamma = 0;
    optionsRHMsdf.method = 'RH-con-fixedstep';
    optionsRHMsdf.maxiter = maxiter;
    [~,~,info_rhg_sdf,~] = rhm(problem, xy0, optionsRHMsdf);
    
    
    % RHM-con-fixedstep
    optionsRHMcon.stepsize = 10;
    optionsRHMcon.gamma = 0.5;
    optionsRHMcon.method = 'RH-con-fixedstep';
    optionsRHMcon.maxiter = maxiter;
    [~,~,info_rhg_con,~] = rhm(problem, xy0, optionsRHMcon);
    
    
    optionsRHMcg.method = 'RH-CG';
    optionsRHMcg.maxiter = maxiter;
    [~,~,info_rhg_cg,~] = rhm(problem, xy0, optionsRHMcg);
    

    % RGDA
    optionsGDA.stepsize = 6;
    optionsGDA.maxiter = maxiter;
    optionsGDA.update = 'retr';
    [~,~,info_gda,~] = rgda(problem, xy0, optionsGDA);
    
    
    %% plots
    colors = {[55, 126, 184]/255, [228, 26, 28]/255, [247, 129, 191]/255, ...
              [166, 86, 40]/255, [255, 255, 51]/255, [255, 127, 0]/255, ...
              [152, 78, 163]/255, [1, 163, 104]/255};

    lw = 1.3;
    
    gda_iter = [info_gda.iter];
    gda_time = [info_gda.time];
    gda_gradnorm = [info_gda.gradnorm];
        
    rhgsdf_iter = [info_rhg_sdf.iter];
    rhgsdf_time = [info_rhg_sdf.time];
    rhgsdf_gradnorm = [info_rhg_sdf.gradnormf];
    
    rhgcon_iter = [info_rhg_con.iter];
    rhgcon_time = [info_rhg_con.time];
    rhgcon_gradnorm = [info_rhg_con.gradnormf];
    
    rhgcg_iter = [info_rhg_cg.iter];
    rhgcg_time = [info_rhg_cg.time];
    rhgcg_gradnorm = [info_rhg_cg.gradnormf];

    % iter
    h1 = figure(1);
    semilogy(gda_iter(1:10:end), gda_gradnorm(1:10:end), '-*', 'color', colors{2}, 'LineWidth',lw); hold on;
    semilogy(rhgsdf_iter(1:10:end), rhgsdf_gradnorm(1:10:end), '-x', 'color', colors{6}, 'LineWidth',lw); hold on;
    semilogy(rhgcon_iter(1:10:end), rhgcon_gradnorm(1:10:end), '-+', 'color', colors{3}, 'LineWidth',lw); hold on;
    semilogy(rhgcg_iter(1:10:end), rhgcg_gradnorm(1:10:end), '-d', 'color', colors{8}, 'LineWidth',lw); hold on;
    hold off;
    ax = gca;
    lg = legend('RGDA', 'RHM-SD-F', 'RHM-CON', 'RHM-CG');
    lg.FontSize = 14;
    xlabel(ax,'Iteration','FontSize',22);
    ylabel(ax,'Gradnorm','FontSize',22);
    

    % time
    h2 = figure(2);
    semilogy(gda_time(1:10:end), gda_gradnorm(1:10:end), '-*', 'color', colors{2}, 'LineWidth',lw); hold on;
    semilogy(rhgsdf_time(1:10:end), rhgsdf_gradnorm(1:10:end), '-x', 'color', colors{6}, 'LineWidth',lw); hold on;
    semilogy(rhgcon_time(1:10:end), rhgcon_gradnorm(1:10:end), '-+', 'color', colors{3}, 'LineWidth',lw); hold on;
    semilogy(rhgcg_time(1:10:end), rhgcg_gradnorm(1:10:end), '-d', 'color', colors{8}, 'LineWidth',lw); hold on;
    hold off;
    ax = gca;
    lg = legend('RGDA','RHM-SD-F', 'RHM-CON', 'RHM-CG');
    lg.FontSize = 14;
    xlim
    xlabel(ax,'Time (s)','FontSize',22);
    ylabel(ax,'Gradnorm','FontSize',22);

end
