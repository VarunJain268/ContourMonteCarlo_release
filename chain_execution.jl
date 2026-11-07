# Run run_chain(chain_index) for each chain, with at most chain_workers running at once.
# Track gradient evaluations per chain and add them to the caller's total.
# On failure, wait for active chains before raising the error.

const CHAIN_GRAD_COUNTER_KEY = :contour_monte_carlo_gradient_counter

@inline function target_grad_counter()
    return get!(() -> Ref(0), task_local_storage(), CHAIN_GRAD_COUNTER_KEY)::Base.RefValue{Int}
end

@inline function increment_target_grad_counter!()
    return target_grad_counter()[] += 1
end

function reset_target_grad_counter!()
    return target_grad_counter()[] = 0
end

function read_target_grad_counter()
    return target_grad_counter()[]
end

function validate_chain_workers(chain_workers::Int)
    available = Threads.nthreads(:default)
    1 <= chain_workers <= available || throw(ArgumentError(
        "chain_workers must be in 1:$available; start Julia with --threads=N to use N workers."))
    return chain_workers
end

# Called for every target before running chains; ordinary targets need no extra check.
# PosteriorDB targets override this to require Stan threading support for parallel runs.
check_chain_target(target, chain_workers::Int) = nothing

function run_chain_jobs(run_chain, n_chains::Int; chain_workers::Int)
    # Validate inputs and prepare per-chain counts.
    validate_chain_workers(chain_workers)
    n_chains >= 0 || throw(ArgumentError("n_chains must be nonnegative."))
    gradient_counts = zeros(Int, n_chains)
    parent_counter = target_grad_counter()

    # Temporarily give this chain its own counter.
    function run_job(chain_index)
        counter = Ref(0)
        task_local_storage(CHAIN_GRAD_COUNTER_KEY, counter) do
            try
                run_chain(chain_index)
            finally
                gradient_counts[chain_index] = counter[]
            end
        end
    end

    try
        if chain_workers == 1
            # Run chains sequentially in the calling task.
            for chain_index in 1:n_chains
                run_job(chain_index)
            end
        else
            # Each worker claims the next chain until none remain or one fails.
            next_chain = Threads.Atomic{Int}(1)
            failed = Threads.Atomic{Bool}(false)
            errors = Vector{Any}(nothing, n_chains)
            @sync for _ in 1:min(chain_workers, n_chains)
                Threads.@spawn begin
                    while !failed[]
                        chain_index = Threads.atomic_add!(next_chain, 1)
                        chain_index > n_chains && break
                        try
                            run_job(chain_index)
                        catch err
                            errors[chain_index] = err
                            failed[] = true
                        end
                    end
                end
            end
            first_error = findfirst(!isnothing, errors)
            first_error === nothing || throw(errors[first_error])
        end
    finally
        # Include completed work even if a chain raised an error.
        parent_counter[] += sum(gradient_counts)
    end
    return gradient_counts
end
