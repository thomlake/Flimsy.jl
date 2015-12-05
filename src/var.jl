abstract AbstractVariable

abstract Variable{T,M,N} <: AbstractVariable

immutable NativeVariable{T<:AbstractMatrix,M,N} <: Variable{T,M,N}
    data::T
    grad::T
end

NativeVariable{T<:AbstractMatrix}(d::T, g::T) = NativeVariable{T,size(d,1),size(d,2)}(d, g)

Variable(x::AbstractMatrix) = NativeVariable(x, zero(x))

Variable(x::AbstractVector) = NativeVariable(reshape(x, length(x), 1), reshape(zero(x), length(x), 1))

Base.eltype{T,M,N}(::Type{Variable{T,M,N}}) = eltype(T)

Base.eltype{T<:AbstractVariable}(x::T) = eltype(T)

Base.size{T,M,N}(::Type{Variable{T,M,N}}) = (M, N)

Base.size(x::Variable) = size(x.data)

Base.size(x::Variable, d::Integer) = size(x.data, d)

Base.zeros{T}(::Type{Variable{T}}, m::Integer, n::Integer) = Variable(zeros(T, m, n))

Base.zeros(::Type{Variable}, m::Integer, n::Integer) = Variable(zeros(m, n))

Base.zeros{T}(::Type{Variable{T}}, n::Integer) = Variable(zeros(T, n))

Base.zeros(::Type{Variable}, n::Int) = Variable(zeros(n))

Base.zero(x::Variable) = Variable(zero(x.data))

Base.eachindex(x::Variable) = eachindex(x.data)
