"""
title: Unitary Evolution Recurrent Neural Networks
paper: http://arxiv.org/pdf/1511.06464v2.pdf
"""
immutable SequenceCopyTask
    dim::Int
    prefix_range::UnitRange{Int}
    suffix_range::UnitRange{Int}
    function SequenceCopyTask(d, pr, sr)
        d > 2 || error("d must be greater than 2")
        (last(pr) >= first(pr) && last(pr) > 0) || error("not enough prefix timesteps")
        (last(sr) >= first(sr) && last(sr) > 0) || error("not enough suffix timesteps")
        return new(d, pr, sr)
    end
end

Base.length(copytask::SequenceCopyTask) = copytask.dim

SequenceCopyTask(t::Int) = SequenceCopyTask(10, 10:10, t:t)

# function Base.rand(copytask::SequenceCopyTask)
#     t1 = rand(copytask.prefix_range)
#     t2 = rand(copytask.suffix_range)
#     prefix = rand(1:copytask.dim - 2, t1)
#     suffix = vcat(copytask.dim - 1, rand(1:copytask.dim - 2, t2 - 2), copytask.dim, fill(copytask.dim - 1, t1))
#     xs = vcat(prefix, suffix)
#     ys = vcat(fill(copytask.dim - 1, t1 + t2), prefix)
#     return [Flimsy.Extras.onehot(x, copytask.dim) for x in xs], ys
# end

function Base.rand(copytask::SequenceCopyTask)
    t1 = rand(copytask.prefix_range)
    t2 = rand(copytask.suffix_range)
    
    prefix = rand(1:copytask.dim - 2, t1)   # numbers to remember
    blanks = fill(copytask.dim - 1, t2 - 1) # blanks
    flag = copytask.dim                     # flag indicating recall should begin
    suffix = fill(copytask.dim - 1, t1)     # blanks
    
    xs = vcat(prefix, blanks, flag, suffix)
    ys = vcat(fill(copytask.dim - 1, t1 + t2), prefix)
    
    return [Flimsy.Extras.onehot(x, copytask.dim) for x in xs], ys
end

function Base.rand(copytask::SequenceCopyTask, n::Int)
    x, y = rand(copytask)
    X, Y = typeof(x)[x], typeof(y)[y]
    for i = 2:n
        x, y = rand(copytask)
        push!(X, x)
        push!(Y, y)
    end
    X, Y
end

