
type ReverseDecat{T<:GradVariable} <: ReverseOperation
    ys::Vector{T}
    x::T
end

function call(rop::ReverseDecat)
    ys = rop.ys
    x = rop.x
    m, n = size(x)
    for j = 1:n
        for i = 1:m
            x.grad[i,j] += ys[i].grad[j]
        end
    end
    return nothing
end

decat(x::AbstractArray) = [x[i,:] for i = 1:size(x, 1)]

decat(x::Variable) = map(DataVariable, decat(x.data))

decat(stack::CallbackStack, x::DataVariable) = decat(x)

function decat(stack::CallbackStack, x::GradVariable)
    ys = map(GradVariable, decat(x.data))
    push!(stack, ReverseDecat(ys, x))
    return ys
end
