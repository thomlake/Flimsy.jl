
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
    curr = heap.curr
    heap.curr = curr + s
    if heap.curr > heap.tail
        throw(HeapFullError(s, heap.tail - curr))
    end
    return pointer_to_array(Ptr{t}(curr), dims)
end

function reset!(heap)
    heap.curr = heap.head
    return heap
end

typealias CallbackStack Array{ReverseOperation,1}

# --------------- #
# Base Scope Type #
# --------------- #
abstract Scope{P}

Base.similar(scope::Scope, x::AbstractArray, initial_value::Real) = fill!(similar(scope, x), initial_value)

allocate(scope::Scope, T::DataType, sz, initial_value::Real) = fill!(allocate(scope, T, sz), initial_value)

# -------------- #
# Gradient Scope #
# -------------- #
abstract GradScope{P} <: Scope{P}

function backprop!(scope::GradScope)
    stack = scope.stack
    for i = endof(stack):-1:1
        stack[i]()
    end
    empty!(stack)
end

function gradient!(f::Function, scope::GradScope, args...)
    y = f(scope, args...)
    backprop!(stack)
    reset!(scope)
    return y
end

function push_callback!(scope::GradScope, cb::ReverseOperation)
    push!(scope.stack, cb)
end

# ------------- #
# Dynamic Scope #
# ------------- #
type DynamicScope{P<:Component} <: Scope{P}
    params::P
end

DynamicScope() = DynamicScope(Components.EmptyComponent())

Base.similar(scope::DynamicScope, x::AbstractArray) = similar(x)

allocate(scope::DynamicScope, T::DataType, sz) = zeros(T, sz)

# ---------------------- #
# Dynamic Gradient Scope #
# ---------------------- #
type DynamicGradScope{P<:Component} <: GradScope{P}
    params::P
    stack::CallbackStack
end

GradScope(scope::DynamicScope) = DynamicGradScope(scope.params, CallbackStack())

Base.similar(scope::DynamicGradScope, x::AbstractArray) = similar(x)

allocate(scope::DynamicGradScope, T::DataType, sz) = zeros(T, sz)

reset!(scope::DynamicGradScope) = nothing

# ------------ #
# Static Scope #
# ------------ #
type StaticScope{P<:Component} <: Scope{P}
    params::P
    heap::Heap
end

StaticScope(params::Component, sz::Int=FLIMSY_DEFAULT_HEAP_SIZE) = StaticScope(params, Heap(sz))

StaticScope(sz::Int=FLIMSY_DEFAULT_HEAP_SIZE) = StaticScope(Components.EmptyComponent(), sz)

Scope(params::Component, sz::Int=FLIMSY_DEFAULT_HEAP_SIZE) = StaticScope(params, sz)

Base.similar(scope::StaticScope, x::AbstractArray) = scope.heap(eltype(x), size(x))

allocate(scope::StaticScope, T::DataType, sz) = scope.heap(T, sz)

reset!(scope::StaticScope) = reset!(scope.heap)

# --------------------- #
# Static Gradient Scope #
# --------------------- #
type StaticGradScope{P<:Component} <: GradScope{P}
    params::P
    heap::Heap
    stack::CallbackStack
end

GradScope(scope::StaticScope) = StaticGradScope(scope.params, scope.heap, CallbackStack())

Base.similar(scope::StaticGradScope, x::AbstractArray) = scope.heap(eltype(x), size(x))

allocate(scope::StaticGradScope, T::DataType, sz) = scope.heap(T, sz)

reset!(scope::StaticGradScope) = reset!(scope.heap)

