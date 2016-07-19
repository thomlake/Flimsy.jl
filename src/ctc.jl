"""
Support fuctions for Connectionist Temporal Classification (CTC) cost function.
"""
module Ctc

using .. Flimsy
using Iterators

allseqs(symbols::Vector{Int}, T::Int) = map(collect, product(repeated(symbols, T)...))

"""
Return a vector of length 2 * length(xs) + 1 with the blank 
symbol added to the start, end, and between each entry in xs.

`expand{I<:Integer}(xs::Vector{I}, blank::Int)`

*Usage*

`expand([2,3,3,4], 1) => [1,2,1,3,1,3,1,4,1]`
"""
function Base.expand{I<:Integer}(xs::Vector{I}, blank::Int)
    ys = fill(blank, 2 * length(xs) + 1)
    for t = 1:length(xs)
        ys[2 * t] = xs[t]
    end
    return ys
end

function trim(xs::Vector{Int}, blank::Int)
    ys = Int[xs[1]]
    for t = 2:length(xs)
        if xs[t] != ys[end]
            push!(ys, xs[t])
        end
    end
    return filter(y -> y != blank, ys)
end

triminv(xs::Vector{Int}, T::Int, symbols::Vector{Int}, blank::Int) = filter(ys -> trim(ys, blank) == xs, allseqs(symbols, T))

function bruteforce(xs::Vector{Int}, lpmat::Matrix, symbols::Vector{Int}, blank::Int)
    T = size(lpmat, 2)
    lpsum = -Inf
    for ys in triminv(xs, T, symbols, blank)
        lp = lpmat[ys[1], 1]
        for t = 2:T
            lp += lpmat[ys[t], t]
        end
        lpsum = Flimsy.Extras.logsumexp(lpsum, lp)
    end
    return lpsum
end

type CtcError <: Exception
    dsc::ASCIIString
    msg::ASCIIString
end

Base.showerror(io::IO, e::CtcError) = print(io, "CtcError(", e.dsc, "): ", e.msg)

type DPTable
    matrix::Matrix{FloatX}
end

DPTable(n_symbols::Int, n_timesteps::Int) = DPTable(fill(-Inf, n_symbols, n_timesteps))

Base.size(table::DPTable) = size(table.matrix)

Base.size(table::DPTable, d::Int) = size(table.matrix, d)

Base.ndims(table::DPTable) = 2

function Base.getindex(table::DPTable, i::Int, j::Int)
    m, n = size(table)
    0 < i <= m || return -Inf
    0 < j <= n || return -Inf
    @inbounds return table.matrix[i,j]
end

Base.setindex!(table::DPTable, x::AbstractFloat, i::Int, j::Int) = table.matrix[i, j] = x

function forward(xs::Vector{Int}, lpmat::Matrix{FloatX}, blank::Int)
    xs[1] == xs[end] == blank || throw(CTCError("NOT EXPANDED", "sequence must be expanded before running CTC algorithm"))

    T = size(lpmat, 2)
    S = length(xs)
    table = DPTable(S, T)
    table[1, 1] = lpmat[blank, 1]
    table[2, 1] = lpmat[xs[2], 1]
    for t = 2:T
        for s = 1:S
            a = if xs[s] == blank || (s > 2 && xs[s] == xs[s-2])
                Flimsy.Extras.logsumexp(table[s, t-1], table[s-1, t-1])
            else
                Flimsy.Extras.logsumexp(table[s, t-1], table[s-1, t-1], table[s-2, t-1])
            end
            table[s, t] = lpmat[xs[s], t] + a
        end
    end
    return table.matrix
end

function backward(xs::Vector{Int}, lpmat::Matrix{FloatX}, blank::Int)
    xs[1] == xs[end] == blank || throw(CTCError("NOT EXPANDED", "sequence must be expanded before running CTC algorithm"))

    T = size(lpmat, 2)
    S = length(xs)
    table = DPTable(S, T)

    table[end, end] = 0.0
    table[end-1, end] = 0.0

    for t = T-1:-1:1
        for s = S:-1:1
            a_s_tp1 = table[s, t+1] + lpmat[xs[s], t + 1]
            a_sp1_tp1 = s < S ? table[s+1, t+1] + lpmat[xs[s+1], t + 1] : -Inf
            a_sp2_tp1 = s + 1 < S ? table[s+2, t+1] + lpmat[xs[s+2], t + 1] : -Inf
            v = if xs[s] == blank || (s+1 < S && xs[s] == xs[s+2])
                Flimsy.Extras.logsumexp(a_s_tp1, a_sp1_tp1)
            else
                Flimsy.Extras.logsumexp(a_s_tp1, a_sp1_tp1, a_sp2_tp1)
            end
            table[s, t] = v
        end
    end
    return table.matrix
end

function make_lpmat{V<:AbstractValue}(output::Vector{V})
    T = length(output)
    p = [softmax(output[t].data) for t=1:T]
    pmat = hcat([p[t] for t=1:T]...)
    lpmat = log(pmat)
    return lpmat
end

end