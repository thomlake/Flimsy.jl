using Flimsy
using Base.Test

immutable C1 <: Component
    x::AbstractVariable
    y::AbstractVariable
    z::AbstractVariable
    blah::Int
end
C1() = C1(Variable(ones(1)), Variable(2 * ones(2, 2)), Variable(3 * ones(3, 3)), 13)

immutable C2 <: Component
    q::AbstractVariable
    c1::C1
    blah::ASCIIString
end
C2() = C2(Variable([1 4; 2 5; 3 6.0]), C1(), "so it goes")

immutable C3{T<:AbstractVariable} <: Component
    blah::Float64
    xs::Vector{T}
end
C3() = C3(13.3, [Variable(ones(1)), Variable(2 * ones(2, 2)), Variable(3 * ones(3, 3))])

immutable C4 <: Component
    blah::Vector{Int}
    c1s::Vector{C1}
end
C4() = C4([1, 2, 3], [C1() for i = 1:3])

c1 = C1()
params = getparams(c1)
@test length(params) == 3
for k = 1:3
    @test size(params[k]) == (k, k)
    @test all(params[k].data .== k)
end

namedparams = getnamedparams(c1)
@test length(namedparams) == 3
for (k, name) in enumerate([:x, :y, :z])
    @test namedparams[k][1] == name
    @test size(namedparams[k][2]) == (k, k)
    @test all(namedparams[k][2].data .== k)
end

c2 = C2()
params = getparams(c2)
@test length(params) == 4
@test size(params[1]) == (3, 2)
for i in eachindex(params[1])
    @test params[1].data[i] == i
end
for k = 1:3
    @test size(params[k + 1]) == (k, k)
    @test all(params[k + 1].data .== k)
end
namedparams = getnamedparams(c2)
@test length(namedparams) == 4
@test namedparams[1][1] == :q
@test size(namedparams[1][2]) == (3, 2)
for i in eachindex(namedparams[1][2])
    @test namedparams[1][2].data[i] == i
end
for (k, name) in enumerate([:x, :y, :z])
    @test namedparams[k + 1][1] == (:c1, name)
    @test size(namedparams[k + 1][2]) == (k, k)
    @test all(namedparams[k + 1][2].data .== k)
end

c3 = C3()
params = getparams(c3)
@test length(params) == 3
for k = 1:3
    @test size(params[k]) == (k, k)
    @test all(params[k].data .== k)
end
namedparams = getnamedparams(c3)
@test length(namedparams) == 3
for k = 1:3
    @test namedparams[k][1] == (:xs, k)
    @test size(namedparams[k][2]) == (k, k)
    @test all(namedparams[k][2].data .== k)
end

c4 = C4()
params = getparams(c4)
@test length(params) == 9
k = 1
for i = 1:3
    for j = 1:3
        @test size(params[k]) == (j, j)
        @test all(params[k].data .== j)
        k += 1
    end
end
namedparams = getnamedparams(c4)
@test length(namedparams) == 9
k = 1
for i = 1:3
    for (j, name) in enumerate([:x, :y, :z])
        @test namedparams[k][1] == ((:c1s, i), name)
        @test size(namedparams[k][2]) == (j, j)
        @test all(namedparams[k][2].data .== j)
        k += 1
    end
end
