function [xyopt, costopt, Hinfo, options] = rhm(problem, xy0, options)
% Riemannian Hamlitonian gmethods for min-max optimization. 
% The problem is min_x max_y f(x,y)
%
% function [xy, cost, info, options] = rhm(problem)
% function [xy, cost, info, options] = rhm(problem, xy)
% function [xy, cost, info, options] = rhm(problem, xy, options)
% function [xy, cost, info, options] = rhm(problem, [], options)
% 
% The cost is the Hamiltonian

    M = problem.M;
    
    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
                'No cost provided. The algorithm will likely abort.');  
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate gradient is
        % explicitly given in the problem description, as in that case the user
        % seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
               ['No gradient provided. Using an FD approximation instead (slow).\n' ...
                'It may be necessary to increase options.tolgradnorm.\n' ...
                'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end
    if ~canGetHessian(problem) && ~canGetApproxHessian(problem)
        % Note: we do not give a warning if an approximate Hessian is
        % explicitly given in the problem description, as in that case the user
        % seems to be aware of the issue.
        warning('manopt:getHessian:approx', ...
               ['No Hessian provided. Using an FD approximation instead.\n' ...
                'To disable this warning: warning(''off'', ''manopt:getHessian:approx'')']);
        problem.approxhess = approxhessianFD(problem);
    end

    
    % Set local defaults here
    localdefaults.maxiter = 100;
    localdefaults.gamma = 0;
    localdefaults.stepsize = 1e-1; % weight on min-max gradient direction
    localdefaults.method = 'RH-CG'; % 'RH-CG', 'RH-TR', 'RH-SD', 'RH-SD-fixedstep', 'RH-con-fixedstep';
    localdefaults.tolgradnorm = 1e-10; % tolgradnorm for f


    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);

    % set parameters
    stepsize = options.stepsize;
    gamma = options.gamma;
    maxiter = options.maxiter;
    tolgradnorm = options.tolgradnorm; % gradnorm on f
    
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('xy0', 'var') || isempty(xy0)
        xy0 = M.rand();
    end


    % Create the Hamiltonian problem structure
    Hproblem.M = M;
    Hproblem.cost = @Hcost;
    Hproblem.grad = @Hgrad;

    Hoptions.maxiter = maxiter;
    Hoptions.tolgradnorm = 1e-30; % make it ineffective.
    Hoptions.minstepsize = 1e-30; % make it ineffective.
    Hoptions.statsfun = @computestats;
    Hoptions.stopfun = @mystopfun;
    
    function stopnow = mystopfun(problem, xy, info, last)
    	stopnow = (sqrt(2* info(last).cost) < tolgradnorm);
	end

    % statsfun to store the gradnorm of f
    function stats = computestats(Hproblem, xy, stats)
        stats.gradnormf = sqrt(2*stats.cost);
    end

    
    function [f, store] = Hcost(xy, store)
        if ~isfield(store, 'rgrad')
            store.rgrad = getGradient(problem, xy);
        end
        rgrad = store.rgrad;
        f = 0.5*M.norm(xy, rgrad)^2; % cost is hamiltonian
    end


    function [g, store] = Hgrad(xy, store)
        if ~isfield(store, 'rgrad')
            [~, store] = Hcost(xy, store);
        end
        rgrad = store.rgrad;
        
        g = getHessian(problem, xy, rgrad);
    end

    % consensus direction
    function [g, store] = Hgradcon(xy, store)
        if ~isfield(store, 'rgrad')
            [~, store] = Hcost(xy, store);
        end
        rgrad = store.rgrad;

        rgrad_gda.x =  rgrad.x;
        rgrad_gda.y = problem.My.lincomb(xy.y,-1,rgrad.y); 
        
        g = M.lincomb(xy, gamma, rgrad_gda, 1, getHessian(problem, xy, rgrad));
    end


    % main optimization step
    
    if strcmpi(options.method, 'RH-SD') % ignores gamma
        fprintf('Riemannian Hamiltonian steepest descent (ignoring gamma)\n')
        [xyopt, costopt, Hinfo] = steepestdescent(Hproblem, xy0, Hoptions);

    elseif strcmpi(options.method, 'RH-CG') % ignores gamma
        fprintf('Riemannian Hamiltonian conjugate gradient (ignoring gamma)\n')
        [xyopt, costopt, Hinfo] = conjugategradient(Hproblem, xy0, Hoptions);

    elseif strcmpi(options.method, 'RH-TR') % ignores gamma
        fprintf('Riemannian Hamiltonian trustregion (ignoring gamma)\n')
        [xyopt, costopt, Hinfo] = trustregions(Hproblem, xy0, Hoptions);

    elseif strcmpi(options.method, 'RH-SD-fixedstep') % ignores gamma
        fprintf('Riemannian Hamiltonian descent with fixed stepsize (ignoring gamma)\n')
        Hproblem.linesearch = @(xy, dxy) stepsize;
        Hoptions.linesearch = @linesearch_hint; % ls algorithm; not required, will be chosen by default
        Hoptions.ls_backtrack = false; % prevent Armijo backtracking
        Hoptions.ls_force_decrease = false; % accept the step even if it increases the cost
        [xyopt, costopt, Hinfo] = steepestdescent(Hproblem, xy0, Hoptions);

    elseif strcmpi(options.method, 'RH-con-fixedstep') % Takes gamma into account
        fprintf('Riemannian Hamiltonian consensus with fixed stepsize (Gamma = %.1f)\n', gamma)
        Hproblem.grad = @Hgradcon; % Use consensus gradient, requires gamma.
        Hproblem.linesearch = @(xy, dxy) stepsize;
        Hoptions.linesearch = @linesearch_hint; % ls algorithm; not required, will be chosen by default
        Hoptions.ls_backtrack = false; % prevent Armijo backtracking
        Hoptions.ls_force_decrease = false; % accept the step even if it increases the cost
        [xyopt, costopt, Hinfo] = steepestdescent(Hproblem, xy0, Hoptions);   

    else
        error('Method "%s" not applicable for gamma = 0', options.method); 
    end
       
    
    


end

