## AddTask
immutable AddTask
    range::UnitRange{Int}
end

AddTask(t::Int) = AddTask(t:t)

function Base.rand(addtask::AddTask)
    steps = rand(addtask.range)
    i1 = rand(1:steps)
    i2 = i1
    while i1 == i2
        i2 = rand(1:steps)
    end
    n1 = rand(1:10)
    n2 = rand(1:10)
    output = float(n1 + n2)
    input = Vector{Float64}[]

    for i = 1:steps
        x = if i == i1
            [n1, 1]
        elseif i == i2
            [n2, 1]
        else
            [rand(1:10), 0]
        end
        push!(input, x)
    end

    return input, output
end

function Base.rand(addtask::AddTask, n::Int)
    x, y = rand(addtask)
    X, Y = typeof(x)[x], typeof(y)[y]
    for i = 2:n
        x, y = rand(addtask)
        push!(X, x)
        push!(Y, y)
    end
    X, Y
end
