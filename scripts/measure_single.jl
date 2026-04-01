cd(@__DIR__)

using JLD2
using CodecZlib
using Printf

include("../src/modelA.jl")

function op(ϕ)
    (sum(ϕ)/L^2, sum(ϕ.^2))
end

function main()
    @init_state

    thermalize(ϕ, m², L^2)

    maxt = 100L^2
    skip = div(L^2,8)
    mass_id = round(m², digits=3)

    open(joinpath(@__DIR__, "..", "data", "magnetization_L_$(L)_mass_$(mass_id)_id_$(seed).dat"), "w") do io
    for i in 0:maxt
        (M, ϕ2) = op(ϕ)
        Printf.@printf(io, "%i %f %f\n", i, M, ϕ2)

        thermalize(ϕ, m², skip)

        flush(stdout)
    end
    end
end

main()
