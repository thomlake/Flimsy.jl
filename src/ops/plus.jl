"""
Reverse op for elementwise plus.
if      c = plus(a, b) 
where   size(a) == size(b)
then    c[i] = a[i] + b[i].

Gradient propagation
    da[i] += dc[i]
    db[i] += dc[i]
"""
type ReversePlus{A<:AbstractValue,B<:AbstractValue} <: ReverseOperation
    c::Variable
    a::A
    b::B
end

for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    updates = Any[]
    A <: Variable && push!(updates, :(a.grad[i] += c.grad[i]))
    B <: Variable && push!(updates, :(b.grad[i] += c.grad[i]))
    update_block = Expr(:block, updates...)
    defn = :(function call(rop::ReversePlus{$A,$B})
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

function plus_elementwise!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    size(a) == size(b) || throw(DimensionMismatch("size(a) = $(size(a)) != $(size(b)) = size(b)"))
    @flimsy_inbounds for i in eachindex(a)
        c[i] = a[i] + b[i]
    end
    return c
end

plus_elementwise(a::AbstractArray, b::AbstractArray) = plus_elementwise!(similar(a), a, b)

plus(scope::Scope, a::AbstractValue, b::AbstractValue) = Constant(plus_elementwise(a.data, b.data))

for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    defn = :(function plus(scope::GradScope, a::$A, b::$B)
        c = Variable(plus_elementwise(a.data, b.data))
        push_callback!(scope, ReversePlus(c, a, b))
        return c
    end)
    eval(defn)
end


# """
# Reverse op for elementwise plus broadcast over rows.
# if      c = plus(a, b)
# where   size(a) == (1, size(b, 2))
# then    c[i,j] = a[j] + b[i,j]

# Gradient propagation
#     da[j] += dc[i,j]
#     db[i,j] += db[i,j]
# """
# type ReverseRowBroadcastPlus{A<:AbstractValue,B<:AbstractValue} <: ReverseOperation
#     c::GradVariable
#     a::A
#     b::B
# end

# for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
#     updates = Any[]
#     A <: Variable && push!(updates, :(a.grad[j] += c.grad[i,j]))
#     B <: Variable && push!(updates, :(b.grad[i,j] += c.grad[i,j]))
#     update_block = Expr(:block, updates...)
#     defn = :(function call(rop::ReverseRowBroadcastPlus{$A,$B})
#         c = rop.c
#         a = rop.a
#         b = rop.b
#         @flimsy_inbounds for j = 1:size(c, 2)
#             for i = 1:size(c, 1)
#                 $update_block
#             end
#         end
#         nothing
#     end)
#     eval(defn)
# end


# """
# Reverse op for elementwise plus broadcast over columns.
# if      c = plus(a, b)
# where   size(a) == (size(b, 1), 1)
# then    c[i,j] = a[i] + b[i,j]

# Gradient propagation
#     da[i] += dc[i,j]
#     db[i,j] += db[i,j]
# """
# type ReverseColBroadcastPlus{Ta<:Variable,Tb<:Variable} <: ReverseOperation
#     c::GradVariable
#     a::Ta
#     b::Tb
# end

# @generated function call{Ta,Tb}(rop::ReverseColBroadcastPlus{Ta,Tb})
#     updates = Any[]
#     if Ta <: GradVariable
#         push!(updates, :(a.grad[i] += c.grad[i,j]))
#     end
#     if Tb <: GradVariable
#         push!(updates, :(b.grad[i,j] += c.grad[i,j]))
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

# function plus_elementwise!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
#     @flimsy_inbounds for i in eachindex(a)
#         c[i] = a[i] + b[i]
#     end
#     return c
# end

# function plus_row_broadcast!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
#     @flimsy_inbounds for j = 1:size(b, 2)
#         for i = 1:size(b, 1)
#             c[i,j] = a[1,j] + b[i,j]
#         end
#     end
#     return c
# end

# function plus_column_broadcast!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
#     @flimsy_inbounds for j = 1:size(b, 2)
#         for i = 1:size(b, 1)
#             c[i,j] = a[i] + b[i,j]
#         end
#     end
#     return c
# end

# plus(a::AbstractArray, b::AbstractArray) = a .+ b

# @generated function plus{Ta<:Variable,Tb<:Variable}(scope::Scope, a::Ta, b::Tb)
#     if anygrads(Ta, Tb) && scope <: GradScope
#         return quote
#             asz, bsz = size(a), size(b)
#             if asz == bsz
#                 c_data = similar(b.data)
#                 c_grad = zero(c_data)
#                 plus_elementwise!(c_data, a.data, b.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReversePlus(c, a, b))
#                 return c
#             elseif asz == (1, bsz[2])
#                 c_data = similar(b.data)
#                 c_grad = zero(c_data)
#                 plus_row_broadcast!(c_data, a.data, b.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReverseRowBroadcastPlus(c, a, b))
#                 return c
#             elseif asz == (bsz[1], 1)
#                 c_data = similar(b.data)
#                 c_grad = zero(c_data)
#                 plus_column_broadcast!(c_data, a.data, b.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReverseColBroadcastPlus(c, a, b))
#                 return c
#             elseif bsz == (1, asz[2])
#                 c_data = similar(a.data)
#                 c_grad = zero(c_data)
#                 plus_row_broadcast!(c_data, b.data, a.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReverseRowBroadcastPlus(c, b, a))
#                 return c
#             elseif bsz == (asz[1], 1)
#                 c_data = similar(a.data)
#                 c_grad = zero(c_data)
#                 plus_column_broadcast!(c_data, b.data, a.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReverseColBroadcastPlus(c, b, a))
#                 return c
#             else
#                 throw(OperationError("no plus for sizes a: $asz, b: $bsz"))
#             end
#         end
#     else
#         return quote
#             asz, bsz = size(a), size(b)
#             if asz == bsz
#                 c_data = similar(b.data)
#                 plus_elementwise!(c_data, a.data, b.data)
#                 c = DataVariable(c_data)
#                 return c
#             elseif asz == (1, bsz[2])
#                 c_data = similar(b.data)
#                 plus_row_broadcast!(c_data, a.data, b.data)
#                 c = DataVariable(c_data)
#                 return c
#             elseif asz == (bsz[1], 1)
#                 c_data = similar(b.data)
#                 plus_column_broadcast!(c_data, a.data, b.data)
#                 c = DataVariable(c_data)
#                 return c
#             elseif bsz == (1, asz[2])
#                 c_data = similar(a.data)
#                 plus_row_broadcast!(c_data, b.data, a.data)
#                 c = DataVariable(c_data)
#                 return c
#             elseif bsz == (asz[1], 1)
#                 c_data = similar(a.data)
#                 plus_column_broadcast!(c_data, b.data, a.data)
#                 c = DataVariable(c_data)
#                 return c
#             else
#                 throw(OperationError("no plus for sizes a: $asz, b: $bsz"))
#             end
#         end
#     end
# end

# # -- Plus > 2 -- #
# function plus{T<:AbstractArray}(xs::Vector{T})
#     length(xs) == 1 && return xs[1]
#     y = plus(xs[1], xs[2])
#     for i = 3:length(xs)
#         y = plus(y, xs[i])
#     end
#     return y
# end

# function plus{V<:Variable}(scope::Scope, xs::Vector{V})
#     length(xs) == 1 && return xs[1]
#     y = plus(scope, xs[1], xs[2])
#     for i = 3:length(xs)
#         y = plus(scope, y, xs[i])
#     end
#     return y
# end

# function plus(x1, x2, x3, xrest...)
#     y = plus(x1, x2)
#     y = plus(y, x3)
#     for x in xrest
#         y = plus(y, x)
#     end
#     return y
# end

# function plus(scope::Scope, x1, x2, x3, xrest...)
#     y = plus(scope, x1, x2)
#     y = plus(scope, y, x3)
#     for x in xrest
#         y = plus(scope, y, x)
#     end
#     return y
# end

# plus{T<:AbstractArray}(x1::T, x2::T, x3::T, xrest::T...) = plus([x1, x2, x3, xrest...])

# plus{V<:Variable}(scope::Scope, x1::V, x2::V, x3::V, xrest::V...) = plus(scope, [x1, x2, x3, xrest...])
