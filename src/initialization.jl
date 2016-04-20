
"""Glorot (aka Xavier) initialization.
title: Understanding the difficulty of training deep feedforward neural networks
paper: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
"""
function glorot(gain::Real, n_out::Int, n_in::Int)
    s = gain / sqrt(n_out + n_in)
    return rand(Uniform(-s, s), n_out, n_in)
end

glorot(n_out::Int, n_in::Int) = glorot(sqrt(6), n_out, n_in)


"""Orthonormal initialization.
title: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
paper: http://arxiv.org/pdf/1312.6120v3.pdf
"""
function orthonormal end

const DEFAULT_ORTHONORMAL_GAIN = 1.1

const ORTHONORMAL_GAIN_LOOKUP = Dict(
    relu => sqrt(2),
    tanh => 1.1,
    sigmoid => 1.1,
)

function orthonormal(gain::Real, n_out::Int, n_in::Int)
    n = max(n_out, n_in)
    w = gain * svd(randn(n, n))[1]
    return w[1:n_out,1:n_in]
end

orthonormal(f::Function, n_out::Int, n_in::Int) = orthonormal(get(ORTHONORMAL_GAIN_LOOKUP, f, DEFAULT_ORTHONORMAL_GAIN), n_out, n_in)

orthonormal(n_out::Int, n_in::Int) = orthonormal(DEFAULT_ORTHONORMAL_GAIN, n_out, n_in)


"""
    spectral(n_out, n_in[, s])

Init from a normal distribution and scale such that the spectral radius is likely s.
http://danielrapp.github.io/rnn-spectral-radius/
"""
function spectral end

function spectral(n_out::Int, n_in::Int, s::AbstractFloat=1.0)
    n = max(n_out, n_in)
    w = scale!(s / sqrt(n), randn(n, n))
    return w[1:n_out,1:n_in]
end


"""Sparse initialization
title: On the importance of initialization and momentum in deep learning
paper: http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
"""
function Base.sparse(sigma::Real, k::Int, n_out::Int, n_in::Int)
    n_zero = n_in - min(k, n_in)
    w = rand(Normal(0, sigma), n_out, n_in)
    for i = 1:n_out
        J = collect(1:n_in)
        shuffle!(J)
        for j = 1:n_zero
            w[i,J[j]] = 0
        end
    end
    return w
end

Base.sparse(k::Int, n_out::Int, n_in::Int) = sparse(1.0, k, n_out, n_in)

Base.sparse(n_out::Int, n_in::Int) = sparse(1.0, 15, n_out, n_in)
