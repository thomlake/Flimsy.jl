
type HeapFullError <: Exception 
    requested::Int
    available::Int
end

Base.showerror(io::IO, e::HeapFullError) = print(io, "HeapFullError [requested => $(e.requested), available => $(e.available))")

type Heap
    head::Ptr{Void}
    tail::Ptr{Void}
    curr::Ptr{Void}
    function Heap(sz::Int)
        head = Libc.malloc(sz)
        tail = head + sz
        curr = head
        heap = new(head, tail, curr)
        finalizer(heap, heap -> Libc.free(heap.head))
        return heap
    end
end

function call(heap::Heap, t::DataType, dims)
    s = sizeof(t) * prod(dims)
    remain = available(heap)
    if s > remain
        throw(HeapFullError(s, remain))
    end
    p = Ptr{t}(heap.curr)
    heap.curr = heap.curr + s
    return pointer_to_array(p, dims)
end

available(heap::Heap) = heap.tail - heap.curr

function reset!(heap)
    heap.curr = heap.head
    return heap
end

# ------------- #
# CallbackStack #
# ------------- #
typealias CallbackStack Array{ReverseOperation,1}

# --------------- #
# Base Scope Type #
# --------------- #
abstract Scope{P}

Base.similar(scope::Scope, x::AbstractArray, initial_value::Real) = fill!(similar(scope, x), initial_value)

allocate(scope::Scope, T::DataType, sz::Tuple, initial_value::Real) = fill!(allocate(scope, T, sz), initial_value)

similartype{F<:AbstractFloat}(scope::Scope, ::Type{F}) = DataVariable{F}

# -------------- #
# Gradient Scope #
# -------------- #
abstract GradScope <: Scope

similartype{F<:AbstractFloat}(scope::GradScope, ::Type{F}) = GradVariable{F}

function backprop!(scope::GradScope)
    stack = scope.stack
    for i = endof(stack):-1:1
        stack[i]()
    end
    empty!(stack)
end

function gradient!(f::Function, scope::GradScope, args...)
    y = f(scope, args...)
    backprop!(scope)
    reset!(scope)
    return y
end

gradient!(f::Function, scope::Scope, args...) = gradient!(f, GradScope(scope), args...)

function push_callback!(scope::GradScope, cb::ReverseOperation)
    push!(scope.stack, cb)
end

# ------------- #
# Dynamic Scope #
# ------------- #
type DynamicScope <: Scope end

Base.similar(scope::DynamicScope, x::AbstractArray) = similar(x)

allocate(scope::DynamicScope, T::DataType, sz::Tuple) = zeros(T, sz)

available(scope::DynamicScope) = Inf

reset!(scope::DynamicScope) = scope

# ---------------------- #
# Dynamic Gradient Scope #
# ---------------------- #
type DynamicGradScope <: GradScope
    stack::CallbackStack
end

GradScope(scope::DynamicScope) = DynamicGradScope(CallbackStack())

Base.similar(scope::DynamicGradScope, x::AbstractArray) = similar(x)

allocate(scope::DynamicGradScope, T::DataType, sz::Tuple) = zeros(T, sz)

available(scope::DynamicScope) = Inf

reset!(scope::DynamicGradScope) = scope

# ------------ #
# Static Scope #
# ------------ #
type StaticScope <: Scope
    heap::Heap
end

StaticScope(sz::Int=FLIMSY_DEFAULT_HEAP_SIZE) = StaticScope(Heap(sz))

Scope(sz::Int=FLIMSY_DEFAULT_HEAP_SIZE) = StaticScope(sz)

Base.similar(scope::StaticScope, x::AbstractArray) = scope.heap(eltype(x), size(x))

allocate(scope::StaticScope, T::DataType, sz::Tuple) = scope.heap(T, sz)

available(scope::StaticScope) = available(scope.heap)

reset!(scope::StaticScope) = (reset!(scope.heap); scope)

# --------------------- #
# Static Gradient Scope #
# --------------------- #
type StaticGradScope <: GradScope
    heap::Heap
    stack::CallbackStack
end

GradScope(scope::StaticScope) = StaticGradScope(scope.heap, CallbackStack())

Base.similar(scope::StaticGradScope, x::AbstractArray) = scope.heap(eltype(x), size(x))

allocate(scope::StaticGradScope, T::DataType, sz::Tuple) = scope.heap(T, sz)

available(scope::StaticGradScope) = available(scope.heap)

reset!(scope::StaticGradScope) = (reset!(scope.heap); scope)

