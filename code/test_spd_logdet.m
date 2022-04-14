% test on logdet(X)*logdet(Y) where X,Y in SPD(d)
function test_spd_logdet()

    clear
    clc
    rng(99);

    d = 30;    
    
    %% problem define
    problem.Mx = sympositivedefinitefactory(d);
    problem.My = sympositivedefinitefactory(d);
    problem.M = productmanifold(struct('x', problem.Mx, 'y', problem.My));

    problem.cost = @cost;
    problem.grad = @grad;
    problem.hess = @hess;
    
    function f = cost(xy)
        x = xy.x;
        y = xy.y;
        f = log(det(x)) * log(det(y));
    end
    
    function g = grad(xy)
        x = xy.x;
        y = xy.y;
        g.x = x .* log(det(y));
        g.y = y .* log(det(x));
    end
    
    function h = hess(xy, xydot)
        x = xy.x;
        y = xy.y;
        xdot = xydot.x;
        ydot = xydot.y;
    
        h.x = x .* trace( y \ ydot);
        h.y = y .* trace( x \ xdot);
    end

    xy0 = problem.M.rand();

    maxiter = 40;
    maxiter_tr = 20;
    
    %% Riem HGD
    % use spdfactory_exp to use exponential map
    problemHGD.Mx = sympositivedefinitefactory_exp(d);
    problemHGD.My = sympositivedefinitefactory_exp(d);
    problemHGD.M = productmanifold(struct('x', problemHGD.Mx, 'y', problemHGD.My));


    problemHGD.cost = @cost;
    problemHGD.grad = @grad;
    problemHGD.hess = @hess;
    
    
    % RHM-fixedstep
    optionsRHMsdf.stepsize = 1e-3;
    optionsRHMsdf.gamma = 0;
    optionsRHMsdf.method = 'RH-con-fixedstep';
    optionsRHMsdf.maxiter = maxiter;
    [~,~,info_rhg_sdf,~] = rhm(problemHGD, xy0, optionsRHMsdf);
    
    
    % RHM-con-fixedstep
    optionsRHMcon.stepsize = 1e-3;
    optionsRHMcon.gamma = 0.5;
    optionsRHMcon.method = 'RH-con-fixedstep';
    optionsRHMcon.maxiter = maxiter;
    [~,~,info_rhg_con,~] = rhm(problemHGD, xy0, optionsRHMcon);
    
    
    optionsRHMsd.method = 'RH-SD';
    optionsRHMsd.maxiter = maxiter;
    [~,~,info_rhg_sd,~] = rhm(problemHGD, xy0, optionsRHMsd);

    optionsRHMcg.method = 'RH-CG';
    optionsRHMcg.maxiter = maxiter;
    [~,~,info_rhg_cg,~] = rhm(problemHGD, xy0, optionsRHMcg);

    optionsRHMtr.method = 'RH-TR';
    optionsRHMtr.maxiter = maxiter_tr;
    [~,~,info_rhg_tr,~] = rhm(problemHGD, xy0, optionsRHMtr);
    

    
    
    %% RGDA
    
    optionsGDA.stepsize = 5e-3;
    optionsGDA.maxiter = maxiter;
    optionsGDA.update = 'exp';
    [~,~,info_gda,~] = rgda(problem, xy0, optionsGDA);
    %}
    
    
    %% RCEG
    optionsRCEG.stepsize = 2e-2;
    optionsRCEG.gamma = 1;
    optionsRCEG.maxiter = maxiter;
    optionsRCEG.update = 'exp';
    optionsRCEG.logchoice = 'log';
    [~,~,info_rceg,~] = rceg(problem, xy0, optionsRCEG);
    
    
    
    %%
    colors = {[55, 126, 184]/255, [228, 26, 28]/255, [247, 129, 191]/255, ...
              [166, 86, 40]/255, [255, 255, 51]/255, [255, 127, 0]/255, ...
              [152, 78, 163]/255, [1, 163, 104]/255};

    lw = 1.3;

    % iter
    h1 = figure(1);
    semilogy([info_gda.iter], [info_gda.gradnorm], '-*', 'color', colors{2}, 'LineWidth',lw); hold on;
    semilogy([info_rceg.iter], [info_rceg.gradnorm], '-o', 'color', colors{1}, 'LineWidth',lw); hold on;
    semilogy([info_rhg_sdf.iter], [info_rhg_sdf.gradnormf], '-x', 'color', colors{6}, 'LineWidth',lw); hold on;
    semilogy([info_rhg_con.iter], [info_rhg_con.gradnormf], '-+', 'color', colors{3}, 'LineWidth',lw); hold on;
    semilogy([info_rhg_sd.iter], [info_rhg_sd.gradnormf], '-s', 'color', colors{4}, 'LineWidth',lw); hold on;
    semilogy([info_rhg_cg.iter], [info_rhg_cg.gradnormf], '-d', 'color', colors{8}, 'LineWidth',lw); hold on;
    semilogy([info_rhg_tr.iter], [info_rhg_tr.gradnormf], '-^', 'color', colors{7}, 'LineWidth',lw); hold on;
    hold off;
    ax = gca;
    xlim([0, 40]);
    lg = legend('RGDA', 'RCEG', 'RHM-SD-F', 'RHM-CON', 'RHM-SD', 'RHM-CG', 'RHM-TR');
    lg.FontSize = 14;
    xlabel(ax,'Iteration','FontSize',22);
    ylabel(ax,'Gradnorm','FontSize',22);
    

    % time
    h2 = figure(2);
    semilogy([info_gda.time], [info_gda.gradnorm], '-*', 'color', colors{2}, 'LineWidth',lw); hold on;
    semilogy([info_rceg.time], [info_rceg.gradnorm], '-o', 'color', colors{1}, 'LineWidth',lw); hold on;
    semilogy([info_rhg_sdf.time], [info_rhg_sdf.gradnormf], '-x', 'color', colors{6}, 'LineWidth',lw); hold on;
    semilogy([info_rhg_con.time], [info_rhg_con.gradnormf], '-+', 'color', colors{3}, 'LineWidth',lw); hold on;
    semilogy([info_rhg_sd.time], [info_rhg_sd.gradnormf], '-s', 'color', colors{4}, 'LineWidth',lw); hold on;
    semilogy([info_rhg_cg.time], [info_rhg_cg.gradnormf], '-d', 'color', colors{8}, 'LineWidth',lw); hold on;
    semilogy([info_rhg_tr.time], [info_rhg_tr.gradnormf], '-^', 'color', colors{7}, 'LineWidth',lw); hold on;
    hold off;
    ax = gca;
    xlim([0, 0.15]);
    lg = legend('RGDA', 'RCEG', 'RHM-SD-F', 'RHM-CON', 'RHM-SD', 'RHM-CG', 'RHM-TR');
    lg.FontSize = 14;
    xlabel(ax,'Time (s)','FontSize',22);
    ylabel(ax,'Gradnorm','FontSize',22);

end
