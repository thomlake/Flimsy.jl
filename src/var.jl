abstract AbstractVariable

immutable Variable{T<:AbstractMatrix,M,N} <: AbstractVariable
    data::T
    grad::T
end

Variable{T<:AbstractMatrix}(d::T, g::T) = Variable{T,size(d,1),size(d,2)}(d, g)

Variable(x::AbstractMatrix) = Variable(x, zero(x))

Variable(x::AbstractVector) = Variable(reshape(x, length(x), 1), reshape(zero(x), length(x), 1))

Variable(x::Real) = Variable([x])

Base.eltype{T,M,N}(::Type{Variable{T,M,N}}) = eltype(T)

Base.eltype{T<:Variable}(x::T) = eltype(T)

Base.size{T,M,N}(::Type{Variable{T,M,N}}) = (M, N)

Base.size(x::Variable) = size(x.data)

Base.size(x::Variable, d::Integer) = size(x.data, d)

Base.zeros{R<:Number}(::Type{Variable}, ::Type{R}, m::Integer, n::Integer) = Variable(zeros(R, m, n))

Base.zeros(::Type{Variable}, m::Integer, n::Integer) = Variable(zeros(m, n))

Base.zeros{T}(::Type{Variable{T}}, n::Integer) = Variable(zeros(T, n))

Base.zeros(::Type{Variable}, n::Int) = Variable(zeros(n))

Base.zero(x::Variable) = Variable(zero(x.data))

Base.eachindex(x::Variable) = eachindex(x.data)
