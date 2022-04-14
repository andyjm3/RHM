function test_spd_rgpca()
    % test on robust geometry-aware pca (example in arxiv:2202.06950)
    % min_x max_y - x' y x - alpha * E_i[ d^2(y, M_i) ], where x is on sphere
    % y is on SPD, M_i are set of SPD matrices.
    
    clear
    clc
    rng(99);
    
    d = 50;
    n = 40;
    mu = 0.2;
    L = 4.5;

    alpha = 3;

    Ms = zeros(d,d,n);
    Msinv = zeros(d,d,n);

    % generate synthetic dataset
    for ii = 1:n
        temp = randn(d,d);
        [Q,~] = qr(temp);
        sigma = (L - mu).*rand(d,1) + mu; 
        Ms(:,:,ii) = Q * diag(sigma) * Q';
        Msinv(:,:,ii) = Q * diag(1./sigma) * Q';
    end
    
    savedir = './results/';
    
    %% define problem
    problem.Mx = spherefactory(d);
    problem.My = sympositivedefinitefactory(d);
    problem.M = productmanifold(struct('x', problem.Mx, 'y', problem.My));  

    problem.cost = @cost;
    problem.grad = @grad;
    problem.hess = @hess;
    %problem.hess = approxhessianFD(problem);


    symm = @(D) 0.5*(D+D');

    function f = cost(xy)
        x = xy.x;
        y = xy.y;

        f = 0;
        % loss on spd
        for jj = 1:n
            f = f - norm(logm(Msinv(:,:,jj) * y),'fro')^2;
        end
        f = f * (alpha/n);

        % loss on sphere
        f = f - x' * y * x;
    end

    function g = grad(xy)
        x = xy.x;
        y = xy.y;

        g.x = - (eye(d) - x*x') * (2*y*x);% BM: okay
        
        g.y = 0;
        logsum = 0;
        for i = 1 : n
            logsum = logsum - logm(y * Msinv(:, :, i));
        end
        grady = symm(2*logsum*y);
        g.y = (alpha/n) .* grady;
        g.y = g.y -  y * (x*x') * y; % BM: okay 
    end


    function h = hess(xy, xydot)
        x = xy.x;
        y = xy.y;
        xdot = xydot.x;
        ydot = xydot.y;

        hessx = -2*(eye(d) - x*x') * y * xdot + 2 * (x' * y * x) * xdot;
        Dygradx = - (eye(d) - x*x') * (2*ydot*x);
        h.x = hessx + Dygradx;
        
        hessy = 0;
        for i = 1 : n
            temp1 = y * Msinv(:, :, i);
            temp2 = logm(temp1);
            Dygrady = - 2 * temp2 * ydot - 2 * dlogm(temp1, ydot* Msinv(:,:,i)) * y;
            grady = - symm(2 *temp2 * y);
            correction = (ydot) * (y\ grady); % (ydot / y) * grady;
            hessy = hessy + symm(Dygrady) - symm(correction);
        end
        hessy = (alpha/n) .* hessy;
        correction2 = - ydot * (x*x')*y;
        hessy = hessy - 2*symm(ydot * (x*x') * y)  - symm(correction2);
        Dxgrady = - 2*y * symm(xdot*x') * y;
        
        h.y = hessy + Dxgrady;
    end
    %checkgradient(problem);
    %checkhessian(problem);
    %keyboard;


    xy0.x = problem.M.rand().x;
    xy0.y = problem.M.rand().y;
    maxiter = 500;

    %% Riem HGD
    problemHGD.Mx = spherefactory_exp(d);
    problemHGD.My = sympositivedefinitefactory_exp(d);
    problemHGD.M = productmanifold(struct('x',  problemHGD.Mx, 'y', problemHGD.My));
    
    problemHGD.cost = @cost;
    problemHGD.grad = @grad;
    problemHGD.hess = @hess;
    
    
    %% alpha = 3 %%
    
    % RHM-fixedstep
    optionsRHMsdf.stepsize = 0.02;
    optionsRHMsdf.gamma = 0;
    optionsRHMsdf.method = 'RH-con-fixedstep';
    optionsRHMsdf.maxiter = maxiter;
    [~,~,info_rhg_sdf,~] = rhm(problemHGD, xy0, optionsRHMsdf);
        
    % RHM-con-fixedstep
    optionsRHMcon.stepsize = 0.02;
    optionsRHMcon.gamma = 0.5;
    optionsRHMcon.method = 'RH-con-fixedstep';
    optionsRHMcon.maxiter = maxiter;
    [~,~,info_rhg_con,~] = rhm(problemHGD, xy0, optionsRHMcon);

    optionsRHMcg.method = 'RH-CG';
    optionsRHMcg.maxiter = maxiter;
    [~,~,info_rhg_cg,~] = rhm(problemHGD, xy0, optionsRHMcg);    
    
    % RGDA
    optionsGDA.stepsize = 0.2;
    optionsGDA.maxiter = maxiter;
    optionsGDA.update = 'exp';
    [~,~,info_gda,~] = rgda(problem, xy0, optionsGDA);

    % RCEG
    optionsRCEG.stepsize = 0.1;
    optionsRCEG.maxiter = maxiter;
    optionsRCEG.update = 'exp';
    optionsRCEG.logchoice = 'log';
    [~,~,info_rceg,~] = rceg(problem, xy0, optionsRCEG);
    %}
    
    %% alpha = 0.1 %%
    %{
    % RHM-fixedstep
    optionsRHMsdf.stepsize = 0.1;
    optionsRHMsdf.gamma = 0;
    optionsRHMsdf.method = 'RH-con-fixedstep';
    optionsRHMsdf.maxiter = maxiter;
    [~,~,info_rhg_sdf,~] = rhm(problemHGD, xy0, optionsRHMsdf);    
    
    % RHM-con-fixedstep
    optionsRHMcon.stepsize = 0.1;
    optionsRHMcon.gamma = 0.5;
    optionsRHMcon.method = 'RH-con-fixedstep';
    optionsRHMcon.maxiter = maxiter;
    [~,~,info_rhg_con,~] = rhm(problemHGD, xy0, optionsRHMcon);
    
    optionsRHMcg.method = 'RH-CG';
    optionsRHMcg.maxiter = maxiter;
    [~,~,info_rhg_cg,~] = rhm(problemHGD, xy0, optionsRHMcg);    
    
    % RGDA
    optionsGDA.stepsize = 0.05;
    optionsGDA.maxiter = maxiter;
    optionsGDA.update = 'exp';
    [~,~,info_gda,~] = rgda(problem, xy0, optionsGDA);
    
    % RCEG
    optionsRCEG.stepsize = 0.05;
    optionsRCEG.maxiter = maxiter;
    optionsRCEG.update = 'exp';
    optionsRCEG.logchoice = 'log';
    [~,~,info_rceg,~] = rceg(problem, xy0, optionsRCEG);
    %}
    
    %% plots
    colors = {[55, 126, 184]/255, [228, 26, 28]/255, [247, 129, 191]/255, ...
              [166, 86, 40]/255, [255, 255, 51]/255, [255, 127, 0]/255, ...
              [152, 78, 163]/255, [1, 163, 104]/255};

    lw = 1.3;
    
    gda_iter = [info_gda.iter];
    gda_time = [info_gda.time];
    gda_gradnorm = [info_gda.gradnorm];
    
    rceg_iter = [info_rceg.iter];
    rceg_time = [info_rceg.time];
    rceg_gradnorm = [info_rceg.gradnorm];
    
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
    semilogy(gda_iter(1:25:end), gda_gradnorm(1:25:end), '-*', 'color', colors{2}, 'LineWidth',lw); hold on;
    semilogy(rceg_iter(1:25:end), rceg_gradnorm(1:25:end), '-o', 'color', colors{1}, 'LineWidth',lw); hold on;
    semilogy(rhgsdf_iter(1:25:end), rhgsdf_gradnorm(1:25:end), '-x', 'color', colors{6}, 'LineWidth',lw); hold on;
    semilogy(rhgcon_iter(1:25:end), rhgcon_gradnorm(1:25:end), '-+', 'color', colors{3}, 'LineWidth',lw); hold on;
    %semilogy([info_rhg_sd.iter], [info_rhg_sd.gradnormf], '-s', 'color', colors{4}, 'LineWidth',lw); hold on;
    semilogy(rhgcg_iter(1:25:end), rhgcg_gradnorm(1:25:end), '-d', 'color', colors{8}, 'LineWidth',lw); hold on;
    %semilogy([info_rhg_tr.iter], [info_rhg_tr.gradnormf], '-^', 'color', colors{7}, 'LineWidth',lw); hold on;
    hold off;
    ax = gca;
    lg = legend('RGDA', 'RCEG', 'RHM-SD-F', 'RHM-CON', 'RHM-CG');
    lg.FontSize = 14;
    xlabel(ax,'Iteration','FontSize',22);
    ylabel(ax,'Gradnorm','FontSize',22);
    

    % time
    h2 = figure(2);
    semilogy(gda_time(1:25:end), gda_gradnorm(1:25:end), '-*', 'color', colors{2}, 'LineWidth',lw); hold on;
    semilogy(rceg_time(1:25:end), rceg_gradnorm(1:25:end), '-o', 'color', colors{1}, 'LineWidth',lw); hold on;
    semilogy(rhgsdf_time(1:25:end), rhgsdf_gradnorm(1:25:end), '-x', 'color', colors{6}, 'LineWidth',lw); hold on;
    semilogy(rhgcon_time(1:25:end), rhgcon_gradnorm(1:25:end), '-+', 'color', colors{3}, 'LineWidth',lw); hold on;
    %semilogy([info_rhg_sd.time], [info_rhg_sd.gradnormf], '-s', 'color', colors{4}, 'LineWidth',lw); hold on;
    semilogy(rhgcg_time(1:25:end), rhgcg_gradnorm(1:25:end), '-d', 'color', colors{8}, 'LineWidth',lw); hold on;
    %semilogy([info_rhg_tr.time], [info_rhg_tr.gradnormf], '-^', 'color', colors{7}, 'LineWidth',lw); hold on;
    hold off;
    ax = gca;
    lg = legend('RGDA', 'RCEG', 'RHM-SD-F', 'RHM-CON', 'RHM-CG');
    lg.FontSize = 14;
    xlabel(ax,'Time (s)','FontSize',22);
    ylabel(ax,'Gradnorm','FontSize',22);
    
    pdf_print_code(h1, [savedir, 'rgpca_', 'iter_', strrep(num2str(alpha),'.','')]);
    pdf_print_code(h2, [savedir, 'rgpca_', 'time_', strrep(num2str(alpha),'.','')]);
    

end