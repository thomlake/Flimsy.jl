## Sequential xor Data
type XORTask
    range::UnitRange{Int}
end

XORTask(t::Int) = XORTask(t:t)

function Base.rand(xor::XORTask)
    x_bits = rand(0:1, rand(xor.range))
    x = [Flimsy.Extras.onehot(b + 1, 2) for b in x_bits]
    y = (cumsum(x_bits) % 2) + 1
    return x, y
end

function Base.rand(xor::XORTask, n::Int)
    X = Array(Vector{Vector{Float64}}, n)
    Y = Array(Vector{Int}, n)
    for i = 1:n
        x, y = rand(xor)
        X[i] = x
        Y[i] = y
    end
    return X, Y
end