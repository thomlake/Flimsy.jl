"""
Reverse op for elementwise minus.
if      c = minus(a, b) 
where   size(a) == size(b)
then    c[i] = a[i] - b[i].

Gradient propagation
    da[i] += dc[i]
    db[i] -= dc[i]
"""
type ReverseMinus{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseMinus{Tc,Ta,Tb})
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
        @inbounds for i in eachindex(c)
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
type ReverseRowBroadcastMinus{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseRowBroadcastMinus{Tc,Ta,Tb})
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
        @inbounds for j = 1:size(c, 2)
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
type ReverseColBroadcastMinus{Tc<:GradVariable,Ta<:Variable,Tb<:Variable} <: ReverseOperation
    c::Tc
    a::Ta
    b::Tb
end

@generated function call{Tc,Ta,Tb}(rop::ReverseColBroadcastMinus{Tc,Ta,Tb})
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
        @inbounds for j = 1:size(c, 2)
            for i = 1:size(c, 1)
                $inner
            end
        end
        return nothing
    end
end

function minus_elementwise!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    @inbounds for i in eachindex(c)
        c[i] = a[i] - b[i]
    end
    return c
end

function minus_row_broadcast!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    @inbounds for j = 1:size(b, 2)
        for i = 1:size(b, 1)
            c[i,j] = a[1,j] - b[i,j]
        end
    end
    return c
end

function minus_column_broadcast!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    @inbounds for j = 1:size(b, 2)
        for i = 1:size(b, 1)
            c[i,j] = a[i] - b[i,j]
        end
    end
    return c
end

minus(a::AbstractArray, b::AbstractArray) = a .- b

@generated function minus{Ta<:Variable,Tb<:Variable}(scope::Scope, a::Ta, b::Tb)
    if anygrads(Ta, Tb) && scope <: GradScope
        return quote
            asz, bsz = size(a), size(b)
            if asz == bsz
                c_data = similar(scope, b.data)
                minus_elementwise!(c_data, a.data, b.data)
                c = GradVariable(c_data, similar(scope, c_data, 0))
                push_callback!(scope, ReverseMinus(c, a, b))
                return c
            elseif asz == (1, bsz[2])
                c_data = similar(scope, b.data)
                minus_row_broadcast!(c_data, a.data, b.data)
                c = GradVariable(c_data, similar(scope, c_data, 0))
                push_callback!(scope, ReverseRowBroadcastMinus(c, a, b))
                return c
            elseif asz == (bsz[1], 1)
                c_data = similar(scope, b.data)
                minus_column_broadcast!(c_data, a.data, b.data)
                c = GradVariable(c_data, similar(scope, c_data, 0))
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
                c_data = similar(scope, b.data)
                minus_elementwise!(c_data, a.data, b.data)
                c = DataVariable(c_data)
                return c
            elseif asz == (1, bsz[2])
                c_data = similar(scope, b.data)
                minus_row_broadcast!(c_data, a.data, b.data)
                c = DataVariable(c_data)
                return c
            elseif asz == (bsz[1], 1)
                c_data = similar(scope, b.data)
                minus_column_broadcast!(c_data, a.data, b.data)
                c = DataVariable(c_data)
                return c
            else
                throw(OperationError("no minus for sizes a: $asz, b: $bsz"))
            end
        end
    end
end

# @generated function minus{Ta<:Variable,Tb<:Variable}(scope::Scope, a::Ta, b::Tb)
#     if anygrads(Ta, Tb) && scope <: GradScope
#         return quote
#             c = GradVariable(minus(a.data, b.data))
#             asz, bsz = size(a), size(b)
#             if asz == bsz
#                 push!(stack, ReverseMinus(c, a, b))
#             elseif asz == (1, bsz[2])
#                 push!(stack, ReverseRowBroadcastMinus(c, a, b))
#             elseif asz == (bsz[1], 1)
#                 push!(stack, ReverseColBroadcastMinus(c, a, b))
#             else
#                 throw(OperationError("no minus for sizes a: $(size(a)), b: $(size(b))"))
#             end
#             return c
#         end
#     else
#         return :(minus(a, b))
#     end
# end
