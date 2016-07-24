
type IOError <: Exception
    info::ASCIIString
    msg::ASCIIString
end

Base.showerror(io::IO, e::IOError) = print(io, "IOError(", e.info, "): ", e.msg)

# ----------------- #
# Component Reading #
# ----------------- #
function read_type(df::HDF5.DataFile)
    str = read(attrs(df), "type")
    return eval(Main, parse(str))
end


function Base.read{C<:Component}(group::HDF5Group, ::Type{C})
    args = Any[]
    for name in fieldnames(C)
        T = fieldtype(C, name)
        
        if T <: Variable
            data = read(group, string(name))
            push!(args, Variable(data))
        
        elseif T <: Vector && eltype(T) <: Variable
            gvec = group[string(name)]
            len = read(attrs(gvec), "length")
            vals = [read(gvec, string(i)) for i = 1:len]
            vars = Variable[Variable(data) for data in vals]
            push!(args, vars)

        elseif T <: Matrix && eltype(T) <: Variable
            gmat = group[string(name)]
            n_rows = read(attrs(gmat), "n_rows")
            n_cols = read(attrs(gmat), "n_cols")
            dvec = [read(gmat, string(i)) for i = 1:n_rows * n_cols]
            vvec = Variable[Variable(data) for data in dvec]
            push!(args, reshape(vvec, n_rows, n_cols))
        
        elseif T <: Component
            gcomp = group[string(name)]
            c = read(gcomp, read_type(gcomp))
            push!(args, c)
        
        elseif T <: Vector && eltype(T) <: Component
            gcompvec = group[string(name)]
            len = read(attrs(gcompvec), "length")
            cvec = [read(gcompvec["$i"], read_type(gcompvec["$i"])) for i = 1:len]
            push!(args, cvec)

        elseif T <: Matrix && eltype(T) <: Component
            gcompmat = group[string(name)]
            n_rows = read(attrs(gcompmat), "n_rows")
            n_cols = read(attrs(gcompmat), "n_cols")
            cvec = [read(gcompmat["$i"], read_type(gcompmat["$i"])) for i = 1:n_rows * n_cols]
            push!(args, reshape(cvec, n_rows, n_cols))
        
        elseif T <: Activation
            activation_name = read(attrs(group), string(name))
            push!(args, ACTIVATION_LOOKUP[activation_name]())

        elseif T <: Real
            val = read(attrs(group), string(name))
            push!(args, val)

        elseif T <: ASCIIString
            val = read(attrs(group), string(name))
            push!(args, val)
        
        elseif T <: Function
            function_name = read(attrs(group), string(name))
            f = eval(Main, parse(function_name))
            push!(args, f)
        
        else
            throw(IOError(string(C), "default reader does not support type => $name::$T"))
        end
    end
    return C(args...)
end

function restore(f::HDF5File; verbose::Bool=true)
    T = read_type(f)
    timestamp = read(attrs(f), "timestamp")
    dt = DateTime(timestamp)
    if verbose
        println("Reading Component")
        println("  file => ", f)
        println("  type => ", T)
        println("  date => ", dt)
    end
    group = f["params"]
    return read(group, T)
end

function restore(fname::AbstractString; verbose::Bool=true)
    return h5open(fname, "r") do f
        return restore(f; verbose=verbose)
    end
end

# ----------------- #
# Component Writing #
# ----------------- #
function write_type{C<:Component}(::Type{C}, df::HDF5.DataFile)
    attrs(df)["type"] = string(C)
end

function Base.write{C<:Component}(group::HDF5Group, params::C)
    write_type(C, group)
    for name in fieldnames(C)
        T = fieldtype(C, name)
        
        if T <: Variable
            group[string(name)] = getfield(params, name).data
        
        elseif T <: Vector && eltype(T) <: Variable
            gvec = g_create(group, string(name))
            values = getfield(params, name)
            attrs(gvec)["length"] = length(values)
            for i = 1:length(values)
                gvec["$i"] = values[i].data
            end

        elseif T <: Matrix && eltype(T) <: Variable
            gmat = g_create(group, string(name))
            values = getfield(params, name)
            attrs(gmat)["n_rows"] = size(values, 1)
            attrs(gmat)["n_cols"] = size(values, 2)
            for i = 1:length(values)
                gmat["$i"] = values[i].data
            end

        elseif T <: Component
            gcomp = g_create(group, string(name))
            c = write(gcomp, getfield(params, name))
        
        elseif T <: Vector && eltype(T) <: Component
            gcompvec = g_create(group, string(name))
            values = getfield(params, name)
            attrs(gcompvec)["length"] = length(values)
            for i = 1:length(values)
                gcomp = g_create(gcompvec, "$i")
                write(gcomp, values[i])
            end

        elseif T <: Matrix && eltype(T) <: Component
            gcompmat = g_create(group, string(name))
            values = getfield(params, name)
            attrs(gcompmat)["n_rows"] = size(values, 1)
            attrs(gcompmat)["n_cols"] = size(values, 2)
            for i = 1:length(values)
                gcomp = g_create(gcompmat, "$i")
                write(gcomp, values[i])
            end
        
        elseif T <: Activation
            attrs(group)[string(name)] = string(getfield(params, name))

        elseif T <: Real
            attrs(group)[string(name)] = getfield(params, name)
        
        elseif T <: ASCIIString
            attrs(group)[string(name)] = getfield(params, name)

        elseif T <: Function
            attrs(group)[string(name)] = string(getfield(params, name))
        
        else
            throw(IOError(string(C), "default writer does not support type => $name::$T"))
        end
    end
end

function save{C<:Component}(f::HDF5File, params::C)
    timestamp = string(now())
    attrs(f)["timestamp"] = timestamp
    write_type(C, f)
    group = g_create(f, "params")
    write(group, params)
end

function save{C<:Component}(fname::AbstractString, params::C)
    h5open(fname, "w") do f
        save(f, params)
    end
end
