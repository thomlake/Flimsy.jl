
type ReverseDot{Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::GradVariable
    a::Ta
    b::Tb
end

@generated function call{Ta,Tb}(rop::ReverseDot{Ta,Tb})
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[i,j] += c.grad[1,j] * b.data[i,j]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i,j] += c.grad[1,j] * a.data[i,j]))
    end

    inner = Expr(:block, updates...)
    return quote
        c = rop.c
        a = rop.a
        b = rop.b
        m, n = size(a)
        @flimsy_inbounds for j = 1:n
            for i = 1:m
                $inner
            end
        end
        return nothing
    end
end

function dot!(c::Matrix, a::Matrix, b::Matrix)
    m, n = size(a)
    for j = 1:n
        for i = 1:m
            c[1,j] += a[i,j] * b[i,j]
        end
    end
    return c
end

function Base.dot(scope::Scope, a::Variable, b::Variable)
    asz = size(a)
    bsz = size(b)
    asz == bsz || throw(OperationError("no dot for sizes a: $asz, b: $bsz"))
    c_data = zeros(FloatX, 1, asz[2])
    return DataVariable(dot!(c_data, a.data, b.data))
end

@generated function Base.dot{Ta<:Variable,Tb<:Variable}(scope::GradScope, a::Ta, b::Tb)
    if anygrads(Ta, Tb)
        return quote
            asz = size(a)
            bsz = size(b)
            asz == bsz || throw(OperationError("no dot for sizes a: $asz, b: $bsz"))
            c_data = zeros(FloatX, 1, asz[2])
            c = GradVariable(dot!(c_data, a.data, b.data), zero(c_data))
            push_callback!(scope, ReverseDot(c, a, b))
            return c
        end
    else
        return quote
            asz = size(a)
            bsz = size(b)
            asz == bsz || throw(OperationError("no dot for sizes a: $asz, b: $bsz"))
            c_data = zeros(FloatX, 1, asz[2])
            return DataVariable(dot!(c_data, a.data, b.data))
        end
    end
end
