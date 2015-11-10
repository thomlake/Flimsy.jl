abstract AbstractVar

abstract Var{T,M,N} <: AbstractVar

immutable NativeVar{T<:AbstractFloat,M,N} <: Var{T,M,N}
    data::Matrix{T}
    grad::Matrix{T}
end

NativeVar{T}(d::Matrix{T}, g::Matrix{T}) = NativeVar{T,size(d,1),size(d,2)}(d, g)
Var(x::Matrix) = NativeVar(x, zero(x))
Var(x::Vector) = NativeVar(reshape(x, length(x), 1), reshape(zero(x), length(x), 1))

Base.eltype{T,M,N}(::Type{Var{T,M,N}}) = T

Base.eltype{T}(x::Var{T}) = T

Base.size{T,M,N}(::Type{Var{T,M,N}}) = (M, N)

Base.size(x::Var) = size(x.data)
Base.size(x::Var, d::Integer) = size(x.data, d)

Base.zeros{T}(::Type{Var{T}}, m::Integer, n::Integer) = Var(zeros(T, m, n))

Base.zeros(::Type{Var}, m::Integer, n::Integer) = Var(zeros(m, n))

Base.zeros{T}(::Type{Var{T}}, n::Integer) = Var(zeros(T, n))

Base.zeros(::Type{Var}, n::Int) = Var(zeros(n))

Base.zero(x::Var) = Var(zero(x.data))

Base.eachindex(x::Var) = eachindex(x.data)
