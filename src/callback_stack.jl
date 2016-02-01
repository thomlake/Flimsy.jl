
typealias CallbackStack Array{ReverseOperation,1}

function backprop!(stack::CallbackStack)
    for i = endof(stack):-1:1
        stack[i]()
    end
    # empty!(stack)
end

function gradient!(f::Function, args...)
    stack = CallbackStack()
    y = f(stack, args...)
    backprop!(stack)
    return y
end

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

type Scope{P<:Component}
    stack::CallbackStack
    heap::Heap
    params::P
end

Scope(params::Component, sz::Int=10000) = Scope(CallbackStack(), Heap(sz), params)

function Base.similar(scope::Scope, x::AbstractArray)
    return scope.heap(eltype(x), size(x))
end
