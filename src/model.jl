"""
A Flimsy.Model wraps a component and scope in a self contained
type containing information required for running the model, 
computing gradients, managing scope, and running gc.
"""
abstract Model{C}

type DynamicModel{C<:Component} <: Model{C}
    component::C
    scope::DynamicScope
    gradscope::DynamicGradScope
    step::Int
    gc_step::Int
end

Model{C<:Component}(component::C, scope::DynamicScope; gc_step::Int=10) = DynamicModel(component, scope, GradScope(scope), 0, gc_step)

type StaticModel{C<:Component} <: Model{C}
    component::C
    scope::StaticScope
    gradscope::StaticGradScope
    step::Int
    gc_step::Int
end

function Base.show(io::IO, model::StaticModel)
    println(io, "StaticModel with ", available(model.scope.heap), " of ", size(model.scope.heap), " bytes available")
    show(io, model.component, 2)
end

Model{C<:Component}(component::C, scope::StaticScope; gc_step::Int=10) = StaticModel(component, scope, GradScope(scope), 0, gc_step)

function Model{C<:Component}(component::C; heap_size::Int=FLIMSY_DEFAULT_HEAP_SIZE, gc_step::Int=10)
    scope = StaticScope(heap_size)
    gradscope = GradScope(scope)
    return StaticModel(component, scope, gradscope, 0, gc_step)
end


