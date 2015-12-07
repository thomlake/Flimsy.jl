
#-- Normal --#
function Gaussian(n_out::Int, n_in::Int)
    Variable(randn(n_out, n_in))
end

function Gaussian(scale::Real, n_out::Int, n_in::Int)
    Variable(scale * randn(n_out, n_in))
end

#-- Uniform --#
function Uniform(lower::Real, upper::Real, n_out::Int, n_in::Int)
    Variable((upper - lower) .* rand(n_out, n_in) .+ lower)
end

#-- Glorot --#
function Glorot(n_out::Int, n_in::Int)
    g = sqrt(6.0) / sqrt(n_out + n_in)
    Uniform(-g, g, n_out, n_in)
end

#-- Orthonormal --#
function Orthonormal(g::Real, n_out::Int, n_in::Int)
    n = max(n_out, n_in)
    w = g .* svd(randn(n, n))[1]
    Variable(w[1:n_out,1:n_in])
end

function Orthonormal(n_out::Int, n_in::Int)
    n = max(n_out, n_in)
    w = svd(randn(n, n))[1]
    Variable(w[1:n_out,1:n_in])
end

#-- Sparse -- #
function Sparse(n::Int, n_out::Int, n_in::Int)
    n_zero = n_in - min(n, n_in)
    w = randn(n_out, n_in)
    for i = 1:n_out
        jz = collect(1:n_in)
        shuffle!(jz)
        for j = 1:n_zero
            w[i,jz[j]] = 0
        end
    end
    Variable(w)
end

# -- Identity -- #
Identity(n::Int) = Variable(eye(n))

# -- Zeros -- #
Zeros(args...) = Variable(zeros(args))
