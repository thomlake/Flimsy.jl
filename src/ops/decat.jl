
type ReverseDecat <: ReverseOperation
    ys::Vector{Variable}
    x::Variable
end

function call(rop::ReverseDecat)
    ys = rop.ys
    x = rop.x
    m, n = size(x)
    @flimsy_inbounds for j = 1:n, i = 1:m
        x.grad[i,j] += ys[i].grad[j]
    end
    return nothing
end

function decat!{T<:AbstractMatrix}(ys::Vector{T}, x::T)
    @flimsy_inbounds for j = 1:size(x, 2), i = 1:size(x, 1)
        ys[i][j] = x[i,j]
    end
    return ys
end

decat{T<:AbstractMatrix}(x::T) = T[x[i,:] for i=1:size(x, 1)]

function decat(scope::Scope, x::AbstractValue)
    m, n = size(x)
    ys_data = Matrix{FloatX}[Matrix{FloatX}(1, n) for i = 1:m]
    decat!(ys_data, x.data)
    ys = Constant[Constant(y_data) for y_data in ys_data]
    return ys
end

function decat(scope::GradScope, x::Variable)
    m, n = size(x)
    ys_data = Matrix{FloatX}[Matrix{FloatX}(1, n) for i = 1:m]
    decat!(ys_data, x.data)
    ys = Variable[Variable(y_data, zero(y_data)) for y_data in ys_data]
    push_callback!(scope, ReverseDecat(ys, x))
    return ys
end
