function test_spd_quadratic()
% test on (logdet(x))^2 + c logdet(x)logdet(y) - (logdet(y))^2, which is
% geodesic convex-concave.

    clear
    clc
    rng(99);
    d = 30;
   
    c = 0;

    savedir = './results/';


    %% problem define
    problem.Mx = sympositivedefinitefactory(d);
    problem.My = sympositivedefinitefactory(d);
    problem.M = productmanifold(struct('x', problem.Mx, 'y', problem.My));

    problem.cost = @cost;
    problem.egrad = @egrad;
    problem.ehess = @ehess;

    function f = cost(xy)
        x = xy.x;
        y = xy.y;
        logdetx = log(det(x));
        logdety = log(det(y));
        f = logdetx^2 + c * logdetx*logdety - logdety^2;
        f = f/d;
    end

    function g = egrad(xy)
        x = xy.x;
        y = xy.y;
        logdetx = log(det(x));
        logdety = log(det(y));
        xinv = inv(x);
        yinv = inv(y);
        g.x = (2 * logdetx + c *logdety) .* xinv;
        g.x = g.x/d;
        g.y = (-2 * logdety + c * logdetx) .* yinv;
        g.y = g.y/d;
    end

    function h = ehess(xy, xydot)
        x = xy.x;
        y = xy.y;
        xdot = xydot.x;
        ydot = xydot.y; 
        logdetx = log(det(x));
        logdety = log(det(y));
        xinv = inv(x);
        yinv = inv(y);
        h.x = - (2*logdetx + c * logdety) .* (x \ xdot / x) + (2 * trace(x \ xdot)) .* xinv ...
                + xinv .* (c * trace(y \ ydot));
        h.x = h.x/d;
        h.y = - (-2*logdety + c *logdetx) .* (y \ ydot /y) - (2 * trace(y \ ydot)) .* yinv ...
                + yinv .* (c * trace(x \ xdot));
        h.y = h.y/d;
    end
    

    xy0 = problem.M.rand();
    maxiter = 40;
    maxiter_tr = 25;

    %% Riem HGD
    problemHGD.Mx = sympositivedefinitefactory_exp(d);
    problemHGD.My = sympositivedefinitefactory_exp(d);
    problemHGD.M = productmanifold(struct('x', problemHGD.Mx, 'y', problemHGD.My));

    problemHGD.cost = @cost;
    problemHGD.egrad = @egrad;
    problemHGD.ehess = @ehess;
    

    
    %% c = 0.1/1 %%
    %{
    % RHM-fixedstep
    optionsRHMsdf.stepsize = 0.2;
    optionsRHMsdf.gamma = 0;
    optionsRHMsdf.method = 'RH-con-fixedstep';
    optionsRHMsdf.maxiter = maxiter;
    [~,~,info_rhg_sdf,~] = rhm(problemHGD, xy0, optionsRHMsdf);
    
    % RHM-con-fixedstep
    optionsRHMcon.stepsize = 0.2;
    optionsRHMcon.gamma = 0.5;
    optionsRHMcon.method = 'RH-con-fixedstep';
    optionsRHMcon.maxiter = maxiter;
    [~,~,info_rhg_con,~] = rhm(problemHGD, xy0, optionsRHMcon);

    
    % for RH + CG
    optionsRHMsd.method = 'RH-SD';
    optionsRHMsd.maxiter = maxiter;
    [~,~,info_rhg_sd,~] = rhm(problemHGD, xy0, optionsRHMsd);

    optionsRHMcg.method = 'RH-CG';
    optionsRHMcg.maxiter = maxiter;
    [~,~,info_rhg_cg,~] = rhm(problemHGD, xy0, optionsRHMcg);

    optionsRHMtr.method = 'RH-TR';
    optionsRHMtr.maxiter = maxiter_tr;
    [~,~,info_rhg_tr,~] = rhm(problemHGD, xy0, optionsRHMtr);
    
    
    % RGDA
    optionsGDA.stepsize = 0.4;
    optionsGDA.maxiter = maxiter;
    optionsGDA.update = 'exp';
    [~,~,info_gda,~] = rgda(problem, xy0, optionsGDA);
    
    
    % RCEG
    optionsRCEG.stepsize = 0.2;
    optionsRCEG.gamma = 1;
    optionsRCEG.maxiter = maxiter;
    optionsRCEG.update = 'exp';
    [~,~,info_rceg,~] = rceg(problem, xy0, optionsRCEG);
    %}



    %% c = 10 %%
    %{
    % RHM-fixedstep
    optionsRHMsdf.stepsize = 0.01;
    optionsRHMsdf.gamma = 0;
    optionsRHMsdf.method = 'RH-con-fixedstep';
    optionsRHMsdf.maxiter = maxiter;
    [~,~,info_rhg_sdf,~] = rhm(problemHGD, xy0, optionsRHMsdf);

    % RHM-con-fixedstep
    optionsRHMcon.stepsize = 0.01;
    optionsRHMcon.gamma = 0.5;
    optionsRHMcon.method = 'RH-con-fixedstep';
    optionsRHMcon.maxiter = maxiter;
    [~,~,info_rhg_con,~] = rhm(problemHGD, xy0, optionsRHMcon);
    
    % for RH + CG
    optionsRHMsd.method = 'RH-SD';
    optionsRHMsd.maxiter = maxiter;
    [~,~,info_rhg_sd,~] = rhm(problemHGD, xy0, optionsRHMsd);

    optionsRHMcg.method = 'RH-CG';
    optionsRHMcg.maxiter = maxiter;
    [~,~,info_rhg_cg,~] = rhm(problemHGD, xy0, optionsRHMcg);

    optionsRHMtr.method = 'RH-TR';
    optionsRHMtr.maxiter = maxiter_tr;
    [~,~,info_rhg_tr,~] = rhm(problemHGD, xy0, optionsRHMtr);
    
    % RGDA
    optionsGDA.stepsize = 0.02;
    optionsGDA.maxiter = maxiter;
    optionsGDA.update = 'exp';
    [~,~,info_gda,~] = rgda(problem, xy0, optionsGDA);
    
     
    % RCEG
    optionsRCEG.stepsize = 0.08;
    optionsRCEG.gamma = 1;
    optionsRCEG.maxiter = maxiter;
    optionsRCEG.update = 'exp';
    [~,~,info_rceg,~] = rceg(problem, xy0, optionsRCEG);  
    %keyboard;
    %}

    %% c = 0 %%
    
    % RHM-fixedstep
    optionsRHMsdf.stepsize = 0.2;
    optionsRHMsdf.gamma = 0;
    optionsRHMsdf.method = 'RH-con-fixedstep';
    optionsRHMsdf.maxiter = maxiter;
    [~,~,info_rhg_sdf,~] = rhm(problemHGD, xy0, optionsRHMsdf);
    
    % RHM-con-fixedstep
    optionsRHMcon.stepsize = 0.2;
    optionsRHMcon.gamma = 0.5;
    optionsRHMcon.method = 'RH-con-fixedstep';
    optionsRHMcon.maxiter = maxiter;
    [~,~,info_rhg_con,~] = rhm(problemHGD, xy0, optionsRHMcon);    

    % for RH + CG
    optionsRHMsd.method = 'RH-SD';
    optionsRHMsd.maxiter = maxiter;
    [~,~,info_rhg_sd,~] = rhm(problemHGD, xy0, optionsRHMsd);

    optionsRHMcg.method = 'RH-CG';
    optionsRHMcg.maxiter = maxiter;
    [~,~,info_rhg_cg,~] = rhm(problemHGD, xy0, optionsRHMcg);

    optionsRHMtr.method = 'RH-TR';
    optionsRHMtr.maxiter = maxiter_tr;
    [~,~,info_rhg_tr,~] = rhm(problemHGD, xy0, optionsRHMtr);
    

   
    % RGDA
    optionsGDA.stepsize = 0.5;
    optionsGDA.maxiter = maxiter;
    optionsGDA.update = 'exp';
    [~,~,info_gda,~] = rgda(problem, xy0, optionsGDA);
    
    
    
    % RCEG
    optionsRCEG.stepsize = 0.2;
    optionsRCEG.maxiter = maxiter;
    optionsRCEG.update = 'exp';
    [~,~,info_rceg,~] = rceg(problem, xy0, optionsRCEG);    
    %}
    
       
    %% plots
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
    xlim([0 30]);
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
    xlim([0 0.15]);
    lg = legend('RGDA', 'RCEG', 'RHM-SD-F', 'RHM-CON', 'RHM-SD', 'RHM-CG', 'RHM-TR');
    lg.FontSize = 14;
    xlabel(ax,'Time (s)','FontSize',22);
    ylabel(ax,'Gradnorm','FontSize',22);
    
    % save plots
    pdf_print_code(h1, [savedir, 'spd_quadratic_', 'iter_', strrep(num2str(c),'.','')]);
    pdf_print_code(h2, [savedir, 'spd_quadratic_', 'time_', strrep(num2str(c),'.','')]);


end

