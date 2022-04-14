function [xy, mycost, info, options] = rgda(problem, xy, options)
% Riemannian gradient descent ascent for min-max optimization. 
% The problem is min_x max_y f(x,y)
%
% function [xy, cost, info, options] = rgda(problem)
% function [xy, cost, info, options] = rgda(problem, xy)
% function [xy, cost, info, options] = rgda(problem, xy, options)
% function [xy, cost, info, options] = rgda(problem, [], options)
% 
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

    % Set local defaults here
    localdefaults.verbosity = 2;
    localdefaults.maxtime = inf;
    localdefaults.maxiter = 100;
    localdefaults.tolgradnorm = 1e-10; % tolgradnorm on f
    localdefaults.stepsize = 0.2;
    localdefaults.update = 'retr';

    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);

    stepsize = options.stepsize;
    algname = 'RGDA';
    
    start_time = tic();

    % If no initial point x is given by the user, generate one at random.
    if ~exist('xy', 'var') || isempty(xy)
        xy = M.rand();
    end

    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Initialize solution
    [mycost, gradfxy] = getCostGrad(problem, xy, storedb, key);
    gradnorm = M.norm(xy, gradfxy);  % sqrt of Hamiltonian


    % savestats
    iter = 0;
    savedstats = 0;
    stats = savestats();
    info(1) = stats;
    savedstats = savedstats + 1;
    preallocate = options.maxiter + 1;
    info(preallocate).iter = [];

    % Display information header for the user.
    if options.verbosity >= 1
        fprintf('\n-------------------------------------------------------\n');
        fprintf('%s:  iter\t               Hamiltonian \t     stepsize\n', algname);
        fprintf('%s:  %5d\t%+.16e\t%.8e\n', algname, iter, gradnorm, stepsize);
    end
    
    elapsed_time = 0;
    for iter = 1:options.maxiter

        % Record start time.
        start_time = tic();

        % compute direction
        gradfxy_mod.x = gradfxy.x;
        gradfxy_mod.y = problem.My.lincomb(xy.y,-1,gradfxy.y);
        update_v = M.lincomb(xy, -1, gradfxy_mod);

        % gradient update
        if strcmp(options.update, 'exp') && isfield(problem.M, 'exp')
            newxy =  problem.M.exp(xy, update_v, stepsize);
        elseif strcmp(options.update, 'retr')
            newxy =  problem.M.retr(xy, update_v, stepsize);
        end
        newkey = storedb.getNewKey(); 

        % Make new key
        xy = newxy;
        key = newkey;

        %compute grad, gradnorm
        [mycost, gradfxy] = getCostGrad(problem, xy, storedb, key);
        gradnorm = problem.M.norm(xy, gradfxy);

        % == stats save ==
        % Log statistics for freshly executed iteration.
        elapsed_time = elapsed_time + toc(start_time);
        stats = savestats();
        info(savedstats+1) = stats;
        savedstats = savedstats + 1;
        
        % Reset timer.
        elapsed_time = 0;

        if options.verbosity >= 1
            fprintf('%s:  %5d\t%+.16e\t%.8e\n', algname, iter, gradnorm, stepsize);
        end
        
        if gradnorm < options.tolgradnorm
            break;
        end

    end



    % Keep only the relevant portion of the info struct-array.
    info = info(1:savedstats); 


    % Helper function to collect statistics 
    function stats = savestats()
        stats.iter = iter;
        if savedstats == 0
            stats.time = toc(start_time);
            stats.stepsize = NaN;
            stats.cost = mycost;
            stats.gradnorm = gradnorm; %hamiltonian
        else
            stats.time = info(savedstats).time + elapsed_time;
            stats.stepsize = stepsize;
            stats.cost = mycost;
            stats.gradnorm = gradnorm;
        end
        stats = applyStatsfun(problem, xy, storedb, key, options, stats);
    end


end

