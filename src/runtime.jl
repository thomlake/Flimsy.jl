"""
Wraps a component in a type containing information required 
for running the model, garbage collection, and computing gradients.
"""
type Runtime{C<:Component}
    component::C
    datascope::DataScope
    gradscope::GradScope
    step::Int
    freq::Int
end

Runtime{C<:Component}(component::C; freq::Int=10) = Runtime(component, DataScope(), GradScope(), 0, freq)

function Base.show(io::IO, runtime::Runtime)
    println(io, "Runtime(step=", runtime.step, ", freq=", runtime.freq, ")")
    show(io, runtime.component, 2)
end

Base.get(runtime::Runtime, args::Symbol...) = foldl((x, f) -> getfield(x, f), runtime.component, args)
