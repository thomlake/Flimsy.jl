
type ReverseDot{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseDot{Tc,Ta,Tb})
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
    # c = rop.c
    # a = rop.a
    # b = rop.b
    # m, n = size(a)
    # for j = 1:n
    #     for i = 1:m
    #         a.grad[i,j] += c.grad[1,j] * b.data[i,j]
    #         b.grad[i,j] += c.grad[1,j] * a.data[i,j]
    #     end
    # end
    # return nothing
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
    c_data = allocate(scope, eltype(a.data), (1, asz[2]), 0)
    return DataVariable(dot!(c_data, a.data, b.data))
end

@generated function Base.dot{Ta<:Variable,Tb<:Variable}(scope::GradScope, a::Ta, b::Tb)
    if anygrads(Ta, Tb)
        return quote
            asz = size(a)
            bsz = size(b)
            asz == bsz || throw(OperationError("no dot for sizes a: $asz, b: $bsz"))
            c_data = allocate(scope, eltype(a.data), (1, asz[2]), 0)
            c = GradVariable(dot!(c_data, a.data, b.data), similar(scope, c_data, 0))
            push_callback!(scope, ReverseDot(c, a, b))
            return c
        end
    else
        return quote
            asz = size(a)
            bsz = size(b)
            asz == bsz || throw(OperationError("no dot for sizes a: $asz, b: $bsz"))
            c_data = allocate(scope, eltype(a.data), (1, asz[2]), 0)
            return DataVariable(dot!(c_data, a.data, b.data))
        end
    end
end
