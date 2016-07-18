
immutable Bidirectional{R<:RecurrentComponent} <: Component
    fwd::R
    bwd::R
end

Bidirectional{R<:RecurrentComponent}(fwd::R, bwd::R) = Bidirectional{R}(fwd, bwd)

Bidirectional{R<:RecurrentComponent}(::Type{R}, m::Int, n::Int) = Bidirectional(R(m, n), R(m, n))

function unfold(scope::Scope, params::Bidirectional, x::Vector)
    y_fwd = unfold(scope, params.fwd, x)
    y_bwd = reverse(unfold(scope, params.bwd, reverse(x)))
    y = [concat(scope, y_fwd[t], y_bwd[t]) for t = 1:length(x)]
    return y
end
