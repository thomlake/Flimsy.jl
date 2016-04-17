"""
Reverse op for elementwise minus.
if      c = minus(a, b) 
where   size(a) == size(b)
then    c[i] = a[i] - b[i].

Gradient propagation
    da[i] += dc[i]
    db[i] -= dc[i]
"""
type ReverseMinus{Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::GradVariable
    a::Ta
    b::Tb
end

@generated function call{Ta,Tb}(rop::ReverseMinus{Ta,Tb})
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[i] += c.grad[i]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i] -= c.grad[i]))
    end
    inner = Expr(:block, updates...)
    return quote
        c = rop.c
        a = rop.a
        b = rop.b
        @flimsy_inbounds for i in eachindex(c)
            $inner
        end
        return nothing
    end
end

"""
Reverse op for elementwise minus broadcast over rows.
if      c = minus(a, b)
where   size(a) == (1, size(b, 2))
then    c[i,j] = a[j] - b[i,j]

Gradient propagation
    da[j] += dc[i,j]
    db[i,j] -= db[i,j]
"""
type ReverseRowBroadcastMinus{Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::GradVariable
    a::Ta
    b::Tb
end

@generated function call{Ta,Tb}(rop::ReverseRowBroadcastMinus{Ta,Tb})
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[j] += c.grad[i,j]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i,j] -= c.grad[i,j]))
    end
    inner = Expr(:block, updates...)
    return quote
        c = rop.c
        a = rop.a
        b = rop.b
        @flimsy_inbounds for j = 1:size(c, 2)
            for i = 1:size(c, 1)
                $inner
            end
        end
        return nothing
    end
end

"""
Reverse op for elementwise minus broadcast over columns.
if      c = minus(a, b)
where   size(a) == (size(b, 1), 1)
then    c[i,j] = a[i] - b[i,j]

Gradient propagation
    da[i] += dc[i,j]
    db[i,j] -= db[i,j]
"""
type ReverseColBroadcastMinus{Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::GradVariable
    a::Ta
    b::Tb
end

@generated function call{Ta,Tb}(rop::ReverseColBroadcastMinus{Ta,Tb})
    updates = Any[]
    if Ta <: GradVariable
        push!(updates, :(a.grad[i] += c.grad[i,j]))
    end
    if Tb <: GradVariable
        push!(updates, :(b.grad[i,j] -= c.grad[i,j]))
    end
    inner = Expr(:block, updates...)
    return quote
        c = rop.c
        a = rop.a
        b = rop.b
        @flimsy_inbounds for j = 1:size(c, 2)
            for i = 1:size(c, 1)
                $inner
            end
        end
        return nothing
    end
end

function minus_elementwise!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    @flimsy_inbounds for i in eachindex(c)
        c[i] = a[i] - b[i]
    end
    return c
end

function minus_row_broadcast!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    @flimsy_inbounds for j = 1:size(b, 2)
        for i = 1:size(b, 1)
            c[i,j] = a[1,j] - b[i,j]
        end
    end
    return c
end

function minus_column_broadcast!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    @flimsy_inbounds for j = 1:size(b, 2)
        for i = 1:size(b, 1)
            c[i,j] = a[i] - b[i,j]
        end
    end
    return c
end

minus(a::AbstractArray, b::AbstractArray) = a .- b

@generated function minus{Ta<:Variable,Tb<:Variable}(scope::Scope, a::Ta, b::Tb)
    if scope <: GradScope && anygrads(Ta, Tb)
        return quote
            asz, bsz = size(a), size(b)
            if asz == bsz
                c_data = similar(b.data)
                c_grad = zero(c_data)
                minus_elementwise!(c_data, a.data, b.data)
                c = GradVariable(c_data, c_grad)
                push_callback!(scope, ReverseMinus(c, a, b))
                return c
            elseif asz == (1, bsz[2])
                c_data = similar(b.data)
                c_grad = zero(c_data)
                minus_row_broadcast!(c_data, a.data, b.data)
                c = GradVariable(c_data, c_grad)
                push_callback!(scope, ReverseRowBroadcastMinus(c, a, b))
                return c
            elseif asz == (bsz[1], 1)
                c_data = similar(b.data)
                c_grad = zero(c_data)
                minus_column_broadcast!(c_data, a.data, b.data)
                c = GradVariable(c_data, c_grad)
                push_callback!(scope, ReverseColBroadcastMinus(c, a, b))
                return c
            else
                throw(OperationError("no minus for sizes a: $asz, b: $bsz"))
            end
            return c
        end
    else
        return quote
            asz, bsz = size(a), size(b)
            if asz == bsz
                c_data = similar(b.data)
                minus_elementwise!(c_data, a.data, b.data)
                c = DataVariable(c_data)
                return c
            elseif asz == (1, bsz[2])
                c_data = similar(b.data)
                minus_row_broadcast!(c_data, a.data, b.data)
                c = DataVariable(c_data)
                return c
            elseif asz == (bsz[1], 1)
                c_data = similar(b.data)
                minus_column_broadcast!(c_data, a.data, b.data)
                c = DataVariable(c_data)
                return c
            else
                throw(OperationError("no minus for sizes a: $asz, b: $bsz"))
            end
        end
    end
end
