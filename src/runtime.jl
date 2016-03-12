"""
Wraps a component and scope in a self contained
type containing information required for running the model, 
computing gradients, managing scope, and running gc.
"""
abstract Runtime{C}

type DynamicRuntime{C<:Component} <: Runtime{C}
    component::C
    scope::DynamicScope
    gradscope::DynamicGradScope
    step::Int
    gc_step::Int
end

function Base.show(io::IO, runtime::DynamicRuntime)
    println(io, "DynamicRuntime")
    show(io, runtime.component, 2)
end

Runtime{C<:Component}(component::C, scope::DynamicScope; gc_step::Int=10) = DynamicRuntime(component, scope, GradScope(scope), 0, gc_step)

type StaticRuntime{C<:Component} <: Runtime{C}
    component::C
    scope::StaticScope
    gradscope::StaticGradScope
    step::Int
    gc_step::Int
end

function Base.show(io::IO, runtime::StaticRuntime)
    println(io, "StaticRuntime with ", available(runtime.scope.heap), " of ", size(runtime.scope.heap), " bytes available")
    show(io, runtime.component, 2)
end

Runtime{C<:Component}(component::C, scope::StaticScope; gc_step::Int=10) = StaticRuntime(component, scope, GradScope(scope), 0, gc_step)

function setup(component::Component; static::Bool=false, heap_size::Int=FLIMSY_DEFAULT_HEAP_SIZE, gc_step::Int=10)
    scope = static ? StaticScope(heap_size): DynamicScope()
    return Runtime(component, scope; gc_step=gc_step)
end

Base.get(runtime::Runtime, args::Symbol...) = foldl((x, f) -> getfield(x, f), runtime.component, args)
