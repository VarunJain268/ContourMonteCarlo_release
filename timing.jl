# Measure runtimes and format timing summaries for console output.

# Execute a block and print elapsed wall time when profiling is enabled.
function timed_exec(label::AbstractString, profile_timing::Bool, f::Function)
    timed = @timed f()
    if profile_timing
        println(">>> Timing: $label took $(round(timed.time, digits=3)) s")
    end
    return timed.value, timed.time
end

# Format seconds for log output.
function fmt_seconds(t)
    return string(round(t, digits=3), " s")
end

# Sum measured runtimes.
function sum_times(times...)
    return sum(Float64.(times))
end