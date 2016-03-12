
type ReverseDecat{T<:GradVariable} <: ReverseOperation
    ys::Vector{T}
    x::T
end

function call(rop::ReverseDecat)
    ys = rop.ys
    x = rop.x
    m, n = size(x)
    @flimsy_inbounds for j = 1:n
        for i = 1:m
            x.grad[i,j] += ys[i].grad[j]
        end
    end
    return nothing
end

function decat!{T}(ys::Vector{Matrix{T}}, x::Matrix{T})
    @flimsy_inbounds for j = 1:size(x, 2)
        for i = 1:size(x, 1)
            ys[i][j] = x[i,j]
        end
    end
    return ys
end

decat(x::Matrix) = map(i -> x[i,:], 1:size(x, 1))

function decat(scope::Scope, x::Variable)
    E = eltype(x)
    m, n = size(x)
    ys_data = Matrix{E}[allocate(scope, E, (1, n)) for i = 1:m]
    decat!(ys_data, x.data)
    ys = DataVariable{E}[DataVariable(y_data) for y_data in ys_data]
    return ys
end

function decat(scope::GradScope, x::GradVariable)
    E = eltype(x)
    m, n = size(x)
    ys_data = Matrix{E}[allocate(scope, E, (1, n)) for i = 1:m]
    decat!(ys_data, x.data)
    ys = GradVariable{E}[GradVariable(y_data, similar(scope, y_data, 0)) for y_data in ys_data]
    push_callback!(scope, ReverseDecat(ys, x))
    return ys
end
