
type ComponentIoError <: Exception
    info::ASCIIString
    msg::ASCIIString
end

Base.showerror(io::IO, e::ComponentIoError) = print(io, "ComponentIoError(", e.info, "): ", e.msg)

# ----------------- #
# Component Reading #
# ----------------- #
function readType(df::HDF5.DataFile)
    str = read(attrs(df), "type")
    return eval(Main, parse(str))
end

function readVariables(names::Vector{Symbol}, group::HDF5Group)
    variables = Any[]
    for name in names
        data = read(group, string(name))
        push!(variables, GradVariable(data, zero(data)))
    end
    return variables
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
            c = read(readType(gcomp), gcomp)
            push!(args, c)
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

function Base.read(f::HDF5File, verbose::Bool=true)
    T = readType(f)
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

function Base.read(fname::ASCIIString, verbose::Bool=true)
    return h5open(fname, "r") do f
        return read(f, verbose)
    end
end

# ----------------- #
# Component Writing #
# ----------------- #
function writeType{C<:Component}(::Type{C}, df::HDF5.DataFile)
    attrs(df)["type"] = string(C)
end

function writeVariables{V<:Variable}(variables::Dict{Symbol,V}, group::HDF5Group)
    for (name, variable) in variables
        group[string(name)] = variable.data
    end
end

function Base.write{C<:Component}(params::C, group::HDF5Group)
    writeType(C, group)
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
        elseif T <: Real
            attrs(group)[string(name)] = getfield(params, name)
        elseif T <: Function
            attrs(group)[string(name)] = string(getfield(params, name))
        else
            throw(ComponentIoError(string(C), "default writer does not support type => $name::$T"))
        end
    end
end

function Base.write{C<:Component}(params::C, f::HDF5File)
    timestamp = string(now())
    attrs(f)["timestamp"] = timestamp
    writeType(C, f)
    group = g_create(f, "params")
    write(params, group)
end

function Base.write{C<:Component}(params::C, fname::ASCIIString)
    h5open(fname, "w") do f
        write(params, f)
    end
end