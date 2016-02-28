
type ComponentIoError <: Exception
    info::ASCIIString
    msg::ASCIIString
end

Base.showerror(io::IO, e::ComponentIoError) = print(io, "ComponentIoError(", e.info, "): ", e.msg)

# ----------------- #
# Component Reading #
# ----------------- #
function read_type(df::HDF5.DataFile)
    str = read(attrs(df), "type")
    return eval(Main, parse(str))
end


function Base.read{C<:Component}(::Type{C}, group::HDF5Group)
    args = Any[]
    for name in fieldnames(C)
        T = fieldtype(C, name)
        if T <: Variable
            data = read(group, string(name))
            push!(args, GradVariable(data, zero(data)))
        elseif T <: Vector && eltype(T) <: Variable
            gvec = group[string(name)]
            len = read(attrs(gvec), "length")
            vals = [read(gvec, string(i)) for i = 1:len]
            vars = GradVariable{eltype(vals[1])}[GradVariable(data, zero(data)) for data in vals]
            push!(args, vars)
        elseif T <: Component
            gcomp = group[string(name)]
            c = read(read_type(gcomp), gcomp)
            push!(args, c)
        elseif T <: Vector && eltype(T) <: Component
            gcompvec = group[string(name)]
            len = read(attrs(gcompvec), "length")
            cvec = [read(read_type(gcompvec["$i"]), gcompvec["$i"]) for i = 1:len]
            push!(args, cvec)
        elseif T <: Real
            val = read(attrs(group), string(name))
            push!(args, val)
        elseif T <: Function
            functionName = read(attrs(group), string(name))
            f = eval(Main, parse(functionName))
            push!(args, f)
        else
            throw(ComponentIoError(string(C), "default reader does not support type => $name::$T"))
        end
    end
    return C(args...)
end

function restore(f::HDF5File, verbose::Bool=true)
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
    return read(T, group)
end

function restore(fname::ASCIIString, verbose::Bool=true)
    return h5open(fname, "r") do f
        return restore(f, verbose)
    end
end

# ----------------- #
# Component Writing #
# ----------------- #
function write_type{C<:Component}(::Type{C}, df::HDF5.DataFile)
    attrs(df)["type"] = string(C)
end

function Base.write{C<:Component}(params::C, group::HDF5Group)
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
        elseif T <: Component
            gcomp = g_create(group, string(name))
            c = write(getfield(params, name), gcomp)
        elseif T <: Vector && eltype(T) <: Component
            gcompvec = g_create(group, string(name))
            values = getfield(params, name)
            attrs(gcompvec)["length"] = length(values)
            for i = 1:length(values)
                gcomp = g_create(gcompvec, "$i")
                write(values[i], gcomp)
            end
        elseif T <: Real
            attrs(group)[string(name)] = getfield(params, name)
        elseif T <: Function
            attrs(group)[string(name)] = string(getfield(params, name))
        else
            throw(ComponentIoError(string(C), "default writer does not support type => $name::$T"))
        end
    end
end

function save{C<:Component}(f::HDF5File, params::C)
    timestamp = string(now())
    attrs(f)["timestamp"] = timestamp
    write_type(C, f)
    group = g_create(f, "params")
    write(params, group)
end

function save{C<:Component}(fname::ASCIIString, params::C)
    h5open(fname, "w") do f
        save(f, params)
    end
end