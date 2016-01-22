
type ReverseDecat{T<:Variable} <: ReverseOperation
    ys::Vector{T}
    x::T
end

call{T<:DataVariable}(rop::ReverseDecat{T}) = nothing

function call{T<:GradVariable}(rop::ReverseDecat{T})
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

decat(x::Variable) = DataVariable{eltype(x)}[DataVariable(x.data[i,:]) for i = 1:size(x, 1)]

function decat(stack::CallbackStack, x::GradVariable)
    ys = GradVariable{eltype(x)}[GradVariable(x.data[i,:]) for i = 1:size(x, 1)]
    push_callback!(stack, ReverseDecat(ys, x))
    return ys
end

decat(stack::CallbackStack, x::DataVariable) = decat(x)
