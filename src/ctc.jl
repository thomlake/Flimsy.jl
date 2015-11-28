using Iterators

allseqs(symbols::Vector{Int}, T::Int) = map(collect, product(repeated(symbols, T)...))

function expand(xs::Vector{Int}, blank::Int)
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

type DPTable
    matrix::Matrix{Float64}
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

Base.setindex!(table::DPTable, x::Float64, i::Int, j::Int) = table.matrix[i, j] = x

function forward(xs::Vector{Int}, lpmat::Matrix{Float64}, blank::Int)
    xs[1] == xs[end] == blank || error("xs must be expanded before running CTC algorithm, try Nimble.CTC.expand(xs)")

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

function backward(xs::Vector{Int}, lpmat::Matrix{Float64}, blank::Int)
    xs[1] == xs[end] == blank || error("xs must be expanded before running CTC algorithm, try Nimble.CTC.expand(xs)")

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

function make_lpmat{V<:Variable}(output::Vector{V}, epsilon::Float64=1e-10)
    T = length(output)
    Pr = [softmax(output[t]) for t=1:T]
    pmat = hcat([Pr[t].data for t=1:T]...)
    lpmat = log(pmat)
    # lpmat = log(pmat .+ epsilon)
    # lpmat = log(max(pmat, epsilon))
    return lpmat
end
