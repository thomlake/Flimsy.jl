using FactCheck
using Flimsy
using Flimsy.Components

immutable A <: Component
    x::Variable
    y::Variable
end

immutable B <: Component
    i1::Int
    i2::Int
    i3::Int
    comp::A
end

immutable C <: Component
    i::Int
    f::Float64
    s::ASCIIString
    varvec::Vector{Variable}
    varmat::Matrix{Variable}
    compvec::Vector{B}
    compmat::Matrix{B}
end

b1 = B(1, 2, 3, A(x=randn(1, 7), y=randn(10, 1)))
b2 = B(4, 5, 6, A(x=randn(1, 1), y=randn(4, 5)))

b11 = B(-1, -2, -3, A(x=randn(1, 1), y=randn(2, 1)))
b21 = B(11, 12, 13, A(x=randn(2, 1), y=randn(1, 2)))
b12 = B(-4, -5, -6, A(x=randn(1, 2), y=randn(2, 2)))
b22 = B(14, 15, 16, A(x=randn(2, 2), y=randn(1, 1)))

c1 = C(
    i=101,
    f=3.14,
    s="the quick brown fox",
    varvec=[randn(3, 6) for i = 1:3],
    varmat=[randn(6, 3) for i = 1:2, j = 1:3],
    compvec=[b1, b2],
    compmat=[b11 b21; b12 b22],
)

path = tempname()
c2 = try
    Flimsy.save(path, c1)
    Flimsy.restore(path, verbose=false)
catch err
    println(err)
end
rm(path)

facts("io") do
    @fact isa(c2, C) --> true
    @fact isa(c2.i, Int) --> true
    @fact c2.i --> c1.i
    @fact isa(c2.f, Float64) --> true
    @fact c2.f --> c1.f
    @fact isa(c2.s, ASCIIString) --> true
    @fact c2.s --> c1.s
    
    @fact isa(c2.varvec, Vector{Variable}) --> true
    @fact length(c2.varvec) --> length(c1.varvec)
    for i = 1:length(c2.varvec)
        @fact isa(c2.varvec[i], Variable) --> true
        @fact size(c2.varvec[i]) --> size(c2.varvec[i])
        @fact all(c2.varvec[i].data .== c1.varvec[i].data) --> true
    end

    @fact isa(c2.varmat, Matrix{Variable}) --> true
    @fact size(c2.varmat) --> size(c1.varmat)
    for j = 1:size(c2.varmat, 2)
        for i = 1:size(c2.varmat, 1)
            @fact isa(c2.varmat[i,j], Variable) --> true
            @fact size(c2.varmat[i,j]) --> size(c2.varmat[i,j])
            @fact all(c2.varmat[i,j].data .== c1.varmat[i,j].data) --> true
        end
    end

    @fact isa(c2.compvec, Vector{B}) --> true
    @fact length(c2.compvec) --> length(c1.compvec)
    for i = 1:length(c2.compvec)
        @fact isa(c2.compvec[i].i1, Int) --> true
        @fact c2.compvec[i].i1 --> c1.compvec[i].i1
        @fact isa(c2.compvec[i].i2, Int) --> true
        @fact c2.compvec[i].i2 --> c1.compvec[i].i2
        @fact isa(c2.compvec[i].i3, Int) --> true
        @fact c2.compvec[i].i3 --> c1.compvec[i].i3
        
        @fact isa(c2.compvec[i].comp, A) --> true
        @fact isa(c2.compvec[i].comp.x, Variable) --> true
        @fact size(c2.compvec[i].comp.x) --> size(c1.compvec[i].comp.x)
        @fact all(c2.compvec[i].comp.x.data .== c1.compvec[i].comp.x.data) --> true
        @fact isa(c2.compvec[i].comp.y, Variable) --> true
        @fact size(c2.compvec[i].comp.y) --> size(c1.compvec[i].comp.y)
        @fact all(c2.compvec[i].comp.y.data .== c1.compvec[i].comp.y.data) --> true
    end
    
    @fact isa(c2.compmat, Matrix{B}) --> true
    @fact size(c2.compmat) --> size(c1.compmat)
    for j = 1:size(c2.compmat, 2)
        for i = 1:size(c2.compmat, 1)
            @fact isa(c2.compmat[i,j].i1, Int) --> true
            @fact c2.compmat[i,j].i1 --> c1.compmat[i,j].i1
            @fact isa(c2.compmat[i,j].i2, Int) --> true
            @fact c2.compmat[i,j].i2 --> c1.compmat[i,j].i2
            @fact isa(c2.compmat[i,j].i3, Int) --> true
            @fact c2.compmat[i,j].i3 --> c1.compmat[i,j].i3
            
            @fact isa(c2.compmat[i,j].comp, A) --> true
            @fact isa(c2.compmat[i,j].comp.x, Variable) --> true
            @fact size(c2.compmat[i,j].comp.x) --> size(c1.compmat[i,j].comp.x)
            @fact all(c2.compmat[i,j].comp.x.data .== c1.compmat[i,j].comp.x.data) --> true
            @fact isa(c2.compmat[i,j].comp.y, Variable) --> true
            @fact size(c2.compmat[i,j].comp.y) --> size(c1.compmat[i,j].comp.y)
            @fact all(c2.compmat[i,j].comp.y.data .== c1.compmat[i,j].comp.y.data) --> true
        end
    end
end

