"""
Reverse op for elementwise minus.
if      c = minus(a, b) 
where   size(a) == size(b)
then    c[i] = a[i] - b[i].

Gradient propagation
    da[i] += dc[i]
    db[i] -= dc[i]
"""
type ReverseMinus{A<:AbstractValue,B<:AbstractValue} <: ReverseOperation
    c::Variable
    a::A
    b::B
end

for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    updates = Any[]
    A <: Variable && push!(updates, :(a.grad[i] += c.grad[i]))
    B <: Variable && push!(updates, :(b.grad[i] -= c.grad[i]))
    update_block = Expr(:block, updates...)
    defn = :(function call(rop::ReverseMinus{$A,$B})
        c = rop.c
        a = rop.a
        b = rop.b
        @flimsy_inbounds for i in eachindex(c)
            $update_block
        end
        nothing
    end)
    eval(defn)
end

function minus_elementwise!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    size(a) == size(b) || throw(DimensionMismatch("size(a) = $(size(a)) != $(size(b)) = size(b)"))
    @flimsy_inbounds for i in eachindex(c)
        c[i] = a[i] - b[i]
    end
    return c
end

minus_elementwise(a::AbstractArray, b::AbstractArray) = minus_elementwise!(similar(a), a, b)

minus(scope::Scope, a::AbstractValue, b::AbstractValue) = Constant(minus_elementwise(a.data, b.data))

for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    defn = :(function minus(scope::GradScope, a::$A, b::$B)
        c = Variable(minus_elementwise(a.data, b.data))
        push_callback!(scope, ReverseMinus(c, a, b))
        return c
    end)
    eval(defn)
end

"""
Reverse op for primative elementwise minus.
if      c = minus(a, b) 
where   isa(a, Real) and isa(b, Variable)
then    c[i] = a - b[i].

Gradient propagation
    db[i] -= dc[i]
"""
type ReverseMinusPrimative{A<:Real} <: ReverseOperation
    c::Variable
    a::A
    b::Variable
end

function call(rop::ReverseMinusPrimative)
    c = rop.c
    b = rop.b
    @flimsy_inbounds for i in eachindex(c)
        b.grad[i] -= c.grad[i]
    end
    nothing
end

function minus_elementwise!(c::AbstractArray, a::Real, b::AbstractArray)
    @flimsy_inbounds for i in eachindex(c)
        c[i] = a - b[i]
    end
    return c
end

minus_elementwise(a::Real, b::AbstractArray) = minus_elementwise!(similar(b), a, b)

minus(scope::Scope, a::Real, b::AbstractValue) = Constant(minus_elementwise(a, b.data))

function minus(scope::GradScope, a::Real, b::Variable)
    c = Variable(minus_elementwise(a, b.data))
    push_callback!(scope, ReverseMinusPrimative(c, a, b))
    return c
end

# """
# Reverse op for elementwise minus broadcast over rows.
# if      c = minus(a, b)
# where   size(a) == (1, size(b, 2))
# then    c[i,j] = a[j] - b[i,j]

# Gradient propagation
#     da[j] += dc[i,j]
#     db[i,j] -= db[i,j]
# """
# type ReverseRowBroadcastMinus{Ta<:Variable,Tb<:Variable} <: ReverseOperation
#     c::GradVariable
#     a::Ta
#     b::Tb
# end

# @generated function call{Ta,Tb}(rop::ReverseRowBroadcastMinus{Ta,Tb})
#     updates = Any[]
#     if Ta <: GradVariable
#         push!(updates, :(a.grad[j] += c.grad[i,j]))
#     end
#     if Tb <: GradVariable
#         push!(updates, :(b.grad[i,j] -= c.grad[i,j]))
#     end
#     inner = Expr(:block, updates...)
#     return quote
#         c = rop.c
#         a = rop.a
#         b = rop.b
#         @flimsy_inbounds for j = 1:size(c, 2)
#             for i = 1:size(c, 1)
#                 $inner
#             end
#         end
#         return nothing
#     end
# end

# """
# Reverse op for elementwise minus broadcast over columns.
# if      c = minus(a, b)
# where   size(a) == (size(b, 1), 1)
# then    c[i,j] = a[i] - b[i,j]

# Gradient propagation
#     da[i] += dc[i,j]
#     db[i,j] -= db[i,j]
# """
# type ReverseColBroadcastMinus{Ta<:Variable,Tb<:Variable} <: ReverseOperation
#     c::GradVariable
#     a::Ta
#     b::Tb
# end

# @generated function call{Ta,Tb}(rop::ReverseColBroadcastMinus{Ta,Tb})
#     updates = Any[]
#     if Ta <: GradVariable
#         push!(updates, :(a.grad[i] += c.grad[i,j]))
#     end
#     if Tb <: GradVariable
#         push!(updates, :(b.grad[i,j] -= c.grad[i,j]))
#     end
#     inner = Expr(:block, updates...)
#     return quote
#         c = rop.c
#         a = rop.a
#         b = rop.b
#         @flimsy_inbounds for j = 1:size(c, 2)
#             for i = 1:size(c, 1)
#                 $inner
#             end
#         end
#         return nothing
#     end
# end

# function minus_elementwise!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
#     @flimsy_inbounds for i in eachindex(c)
#         c[i] = a[i] - b[i]
#     end
#     return c
# end

# function minus_row_broadcast!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
#     @flimsy_inbounds for j = 1:size(b, 2)
#         for i = 1:size(b, 1)
#             c[i,j] = a[1,j] - b[i,j]
#         end
#     end
#     return c
# end

# function minus_column_broadcast!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
#     @flimsy_inbounds for j = 1:size(b, 2)
#         for i = 1:size(b, 1)
#             c[i,j] = a[i] - b[i,j]
#         end
#     end
#     return c
# end

# minus(a::AbstractArray, b::AbstractArray) = a .- b

# @generated function minus{Ta<:Variable,Tb<:Variable}(scope::Scope, a::Ta, b::Tb)
#     if scope <: GradScope && anygrads(Ta, Tb)
#         return quote
#             asz, bsz = size(a), size(b)
#             if asz == bsz
#                 c_data = similar(b.data)
#                 c_grad = zero(c_data)
#                 minus_elementwise!(c_data, a.data, b.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReverseMinus(c, a, b))
#                 return c
#             elseif asz == (1, bsz[2])
#                 c_data = similar(b.data)
#                 c_grad = zero(c_data)
#                 minus_row_broadcast!(c_data, a.data, b.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReverseRowBroadcastMinus(c, a, b))
#                 return c
#             elseif asz == (bsz[1], 1)
#                 c_data = similar(b.data)
#                 c_grad = zero(c_data)
#                 minus_column_broadcast!(c_data, a.data, b.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReverseColBroadcastMinus(c, a, b))
#                 return c
#             else
#                 throw(OperationError("no minus for sizes a: $asz, b: $bsz"))
#             end
#             return c
#         end
#     else
#         return quote
#             asz, bsz = size(a), size(b)
#             if asz == bsz
#                 c_data = similar(b.data)
#                 minus_elementwise!(c_data, a.data, b.data)
#                 c = DataVariable(c_data)
#                 return c
#             elseif asz == (1, bsz[2])
#                 c_data = similar(b.data)
#                 minus_row_broadcast!(c_data, a.data, b.data)
#                 c = DataVariable(c_data)
#                 return c
#             elseif asz == (bsz[1], 1)
#                 c_data = similar(b.data)
#                 minus_column_broadcast!(c_data, a.data, b.data)
#                 c = DataVariable(c_data)
#                 return c
#             else
#                 throw(OperationError("no minus for sizes a: $asz, b: $bsz"))
#             end
#         end
#     end
# end
