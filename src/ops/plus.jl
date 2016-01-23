"""
Reverse op for elementwise plus.
if      c = plus(a, b) 
where   size(a) == size(b)
then    c[i] = a[i] + b[i].

Gradient propagation
    da[i] += dc[i]
    db[i] += dc[i]
"""
type ReversePlus{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReversePlus{Tc,Ta,Tb})
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[i] += c.grad[i]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i] += c.grad[i]))
    end
    inner = Expr(:block, updates...)
    return quote
        c = rop.c
        a = rop.a
        b = rop.b
        for i in eachindex(c)
            $inner
        end
        return nothing
    end
end

"""
Reverse op for elementwise plus broadcast over rows.
if      c = plus(a, b)
where   size(a) == (1, size(b, 2))
then    c[i,j] = a[j] + b[i,j]

Gradient propagation
    da[j] += dc[i,j]
    db[i,j] += db[i,j]
"""
type ReverseRowBroadcastPlus{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseRowBroadcastPlus{Tc,Ta,Tb})
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[j] += c.grad[i,j]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i,j] += c.grad[i,j]))
    end
    inner = Expr(:block, updates...)
    return quote
        c = rop.c
        a = rop.a
        b = rop.b
        for j = 1:size(c, 2)
            for i = 1:size(c, 1)
                $inner
            end
        end
        return nothing
    end
end

"""
Reverse op for elementwise plus broadcast over columns.
if      c = plus(a, b)
where   size(a) == (size(b, 1), 1)
then    c[i,j] = a[i] + b[i,j]

Gradient propagation
    da[i] += dc[i,j]
    db[i,j] += db[i,j]
"""
type ReverseColBroadcastPlus{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseColBroadcastPlus{Tc,Ta,Tb})
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[i] += c.grad[i,j]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i,j] += c.grad[i,j]))
    end
    inner = Expr(:block, updates...)
    return quote
        c = rop.c
        a = rop.a
        b = rop.b
        for j = 1:size(c, 2)
            for i = 1:size(c, 1)
                $inner
            end
        end
        return nothing
    end
end

plus(a::AbstractArray, b::AbstractArray) = a .+ b

plus(a::Variable, b::Variable) = DataVariable(plus(a.data, b.data))

@generated function plus{Ta<:Variable,Tb<:Variable}(stack::CallbackStack, a::Ta, b::Tb)
    if anygrads(Ta, Tb)
        return quote
            c = GradVariable(plus(a.data, b.data))
            asz, bsz = size(a), size(b)
            if asz == bsz
                push!(stack, ReversePlus(c, a, b))
            elseif asz == (1, bsz[2])
                push!(stack, ReverseRowBroadcastPlus(c, a, b))
            elseif asz == (bsz[1], 1)
                push!(stack, ReverseColBroadcastPlus(c, a, b))
            elseif bsz == (1, asz[2])
                push!(stack, ReverseRowBroadcastPlus(c, b, a))
            elseif bsz == (asz[1], 1)
                push!(stack, ReverseColBroadcastPlus(c, b, a))
            else
                throw(OperationError("no plus for sizes a: $(size(a)), b: $(size(b))"))
            end
            return c
        end
    else
        return :(plus(a, b))
    end
end

# -- Plus > 2 -- #
function plus{V<:Variable}(xs::Vector{V})
    y = plus(xs[1], xs[2])
    for i = 3:length(xs)
        y = plus(y, xs[i])
    end
    return y
end

function plus{V<:Variable}(stack::CallbackStack, xs::Vector{V})
    y = plus(stack, xs[1], xs[2])
    for i = 3:length(xs)
        y = plus(stack, y, xs[i])
    end
    return y
end

plus(x1::Variable, x2::Variable, x3::Variable, xrest::Variable...) = plus([x1, x2, x3, xrest...])

plus(stack::CallbackStack, x1::Variable, x2::Variable, x3::Variable, xrest::Variable...) = plus(stack, [x1, x2, x3, xrest...])

