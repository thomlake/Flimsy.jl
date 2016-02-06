
immutable GatedRecurrent{V<:Variable} <: RecurrentComponent{V}
    wr::V; wz::V; wc::V;
    ur::V; uz::V; uc::V;
    br::V; bz::V; bc::V;
    h0::V
    function GatedRecurrent(wr::V, wz::V, wc::V, ur::V, uz::V, uc::V, br::V, bz::V, bc::V, h0::V)
        m, n = size(wr)
        size(wz) == (m, n) || error("Bad size(wz) == $(size(wz)) != ($m, $n)")
        size(wc) == (m, n) || error("Bad size(wc) == $(size(wc)) != ($m, $n)")
        size(ur) == (m, m) || error("Bad size(ur) == $(size(ur)) != ($m, $m)")
        size(uz) == (m, m) || error("Bad size(uz) == $(size(uz)) != ($m, $m)")
        size(uc) == (m, m) || error("Bad size(uc) == $(size(uc)) != ($m, $m)")
        size(br) == (m, 1) || error("Bad size(br) == $(size(br)) != ($m, 1)")
        size(bz) == (m, 1) || error("Bad size(bz) == $(size(bz)) != ($m, 1)")
        size(bc) == (m, 1) || error("Bad size(bc) == $(size(bc)) != ($m, 1)")
        size(h0) == (m, 1) || error("Bad size(h0) == $(size(h0)) != ($m, 1)")
        return new(wr, wz, wc, ur, uz, uc, br, bz, bc, h0)
    end
end

GatedRecurrent(m::Int, n::Int) = GatedRecurrent(
    wr=orthonormal(m, n), wz=orthonormal(m, n), wc=orthonormal(m, n),
    ur=orthonormal(m, m), uz=orthonormal(m, m), uc=orthonormal(m, m),
    br=zeros(m, 1), bz=zeros(m, 1), bc=zeros(m, 1), h0=zeros(m, 1),
)


@component function Base.step(params::GatedRecurrent, x::Variable, htm1::Variable)
    r = sigmoid(plus(linear(params.wr, x), linear(params.ur, htm1), params.br))
    z = sigmoid(plus(linear(params.wz, x), linear(params.uz, htm1), params.bz))
    c = tanh(plus(linear(params.wc, x), mult(r, linear(params.uc, htm1)), params.bc))
    return plus(mult(z, htm1), mult(minus(1.0, z), c))
end

@component Base.step(params::GatedRecurrent, x::Variable) = step(params, x, params.h0)

@component function unfold{T<:Variable}(params::GatedRecurrent, x::Vector{T})
    h = Array(vartype(T), length(x))
    h[1] = step(params, x[1])
    for t = 2:length(x)
        h[t] = step(params, x[t], h[t-1])
    end
    return h
end
