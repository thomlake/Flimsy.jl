
immutable Fusion{V<:Variable} <: Component{V}
    w::Vector{V}
    b::V
    function Fusion(w::Vector{V}, b::V)
        m = size(w[1], 1)
        size(b) == (m, 1) || throw(DimensionMismatch("Bad size(b[1]) != ($m, 1)"))
        for i = 2:length(w)
            n = size(w[i], 1)
            m == n || throw(DimensionMismatch("Bad size(w[$i], 1) == $n != $m"))
        end
        return new(w, b)
    end
end
Fusion{V<:Variable}(w::Vector{V}, b::V) = Fusion{V}(w, b)

function Fusion(n_out::Int, sz::Int...)
    w = [rand(Normal(0, 0.01), n_out, n) for n in sz]
    b = zeros(n_out, 1)
    return Fusion(w=w, b=b)
end

@comp function feedforward(params::Fusion, x::Vector)
    length(x) == length(params.w) || error("Expected ", length(params.w), " inputs, got ", length(x))

    y = params.b
    for i = 1:length(params.w)
        y = plus(y, linear(params.w[i], x[i]))
    end
    return y
end
