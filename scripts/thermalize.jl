cd(@__DIR__)

using JLD2
using CodecZlib

include("../src/modelA.jl")

function main()
    @init_state

    for t in 1:L
        thermalize(ϕ, m², L^2)
        @show t
        flush(stdout)
        jldsave(joinpath(@__DIR__, "..", "data", "thermalized_L_$(L)_id_$(seed).jld2"), true; ϕ=Array(ϕ), m²=m², t=t)
    end
end

main()
