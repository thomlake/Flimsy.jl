"""
Reverse op for elementwise multiplication broadcast over columns.
if      c = mult(a, b)
where   size(b) == (size(a, 1), 1)
then    c[i,j] = a[i,j] * b[i]

Gradient propagation
    da[i,j] += dc[i,j] * b[i]
    db[i] += dc[i,j] * a[i,j]
"""
type ReverseMultCols{A<:AbstractValue,B<:AbstractValue} <: ReverseOperation
    c::Variable
    a::A
    b::B
end

for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    updates = Any[]
    A <: Variable && push!(updates, :(a.grad[i,j] += c.grad[i,j] * b.data[i]))
    B <: Variable && push!(updates, :(b.grad[i] += c.grad[i,j] * a.data[i,j]))
    update_block = Expr(:block, updates...)
    defn = :(function call(rop::ReverseMultCols{$A,$B})
        c = rop.c
        a = rop.a
        b = rop.b
        @flimsy_inbounds for j = 1:size(c, 2), i = 1:size(c, 1)
            $update_block
        end
        nothing
    end)
    eval(defn)
end

function mult_cols!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
    size(b) == (size(a, 1), 1) || throw(DimensionMismatch("b must be $(size(a, 1)) by 1, got $(size(b))"))
    @flimsy_inbounds for j = 1:size(c, 2)
        for i = 1:size(c, 1)
            c[i,j] = a[i,j] * b[i]
        end
    end
    return c
end

mult_cols(a::AbstractArray, b::AbstractArray) = mult_cols!(similar(a), a, b)

mult_cols(scope::Scope, a::AbstractValue, b::AbstractValue) = Constant(mult_cols(a.data, b.data))

for (A, B) in [(Constant,Variable), (Variable,Constant), (Variable,Variable)]
    defn = :(function mult_cols(scope::GradScope, a::$A, b::$B)
        c = Variable(mult_cols(a.data, b.data))
        push_callback!(scope, ReverseMultCols(c, a, b))
        return c
    end)
    eval(defn)
end

# """
# Reverse op for elementwise multiplication broadcast over columns.
# if      c = mult(a, b)
# where   size(a) == (size(b, 1), 1)
# then    c[i,j] = a[i] * b[i,j]

# Gradient propagation
#     da[i] += dc[i,j] * b[i,j]
#     db[i,j] += db[i,j] * a[i]
# """
# type ReverseColBroadcastMult{Ta<:Variable,Tb<:Variable} <: ReverseOperation
#     c::GradVariable
#     a::Ta
#     b::Tb
# end

# @generated function call{Ta,Tb}(rop::ReverseColBroadcastMult{Ta,Tb})
#     updates = Any[]
#     if Ta <: GradVariable
#         push!(updates, :(a.grad[i] += c.grad[i,j] * b.data[i,j]))
#     end
#     if Tb <: GradVariable
#         push!(updates, :(b.grad[i,j] += c.grad[i,j] * a.data[i]))
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

# function mult_elementwise!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
#     @flimsy_inbounds for i in eachindex(a)
#         c[i] = a[i] * b[i]
#     end
#     return c
# end

# function mult_row_broadcast!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
#     @flimsy_inbounds for j = 1:size(b, 2)
#         for i = 1:size(b, 1)
#             c[i,j] = a[1,j] * b[i,j]
#         end
#     end
#     return c
# end

# function mult_column_broadcast!(c::AbstractArray, a::AbstractArray, b::AbstractArray)
#     @flimsy_inbounds for j = 1:size(b, 2)
#         for i = 1:size(b, 1)
#             c[i,j] = a[i] * b[i,j]
#         end
#     end
#     return c
# end

# mult(a::AbstractArray, b::AbstractArray) = a .* b

# mult(a::Variable, b::Variable) = DataVariable(mult(a.data, b.data))

# @generated function mult{Ta<:Variable,Tb<:Variable}(scope::Scope, a::Ta, b::Tb)
#     if anygrads(Ta, Tb) && scope <: GradScope
#         return quote
#             asz, bsz = size(a), size(b)
#             if asz == bsz
#                 c_data = similar(b.data)
#                 c_grad = zero(c_data)
#                 mult_elementwise!(c_data, a.data, b.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReverseMult(c, a, b))
#                 return c
#             elseif asz == (1, bsz[2])
#                 c_data = similar(b.data)
#                 c_grad = zero(c_data)
#                 mult_row_broadcast!(c_data, a.data, b.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReverseRowBroadcastMult(c, a, b))
#                 return c
#             elseif asz == (bsz[1], 1)
#                 c_data = similar(b.data)
#                 c_grad = zero(c_data)
#                 mult_column_broadcast!(c_data, a.data, b.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReverseColBroadcastMult(c, a, b))
#                 return c
#             elseif bsz == (1, asz[2])
#                 c_data = similar(a.data)
#                 c_grad = zero(c_data)
#                 mult_row_broadcast!(c_data, b.data, a.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReverseRowBroadcastMult(c, b, a))
#                 return c
#             elseif bsz == (asz[1], 1)
#                 c_data = similar(a.data)
#                 c_grad = zero(c_data)
#                 mult_column_broadcast!(c_data, b.data, a.data)
#                 c = GradVariable(c_data, c_grad)
#                 push_callback!(scope, ReverseColBroadcastMult(c, b, a))
#                 return c
#             else
#                 throw(OperationError("no mult for sizes a: $(size(a)), b: $(size(b))"))
#             end
#         end
#     else
#         return quote
#             asz, bsz = size(a), size(b)
#             if asz == bsz
#                 c_data = similar(b.data)
#                 mult_elementwise!(c_data, a.data, b.data)
#                 c = DataVariable(c_data)
#                 return c
#             elseif asz == (1, bsz[2])
#                 c_data = similar(b.data)
#                 mult_row_broadcast!(c_data, a.data, b.data)
#                 c = DataVariable(c_data)
#                 return c
#             elseif asz == (bsz[1], 1)
#                 c_data = similar(b.data)
#                 mult_column_broadcast!(c_data, a.data, b.data)
#                 c = DataVariable(c_data)
#                 return c
#             elseif bsz == (1, asz[2])
#                 c_data = similar(a.data)
#                 mult_row_broadcast!(c_data, b.data, a.data)
#                 c = DataVariable(c_data)
#                 return c
#             elseif bsz == (asz[1], 1)
#                 c_data = similar(a.data)
#                 mult_column_broadcast!(c_data, b.data, a.data)
#                 c = DataVariable(c_data)
#                 return c
#             else
#                 throw(OperationError("no mult for sizes a: $(size(a)), b: $(size(b))"))
#             end
#         end
#     end
# end