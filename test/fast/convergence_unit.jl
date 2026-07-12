using Damysos
using Test

import Damysos.currentvalue

@testset "ConvergenceTest unit" begin
    tgrid = SymmetricTimeGrid(0.1, -5.0)
    kgrid = CartesianKGrid1d(0.5, 10.0)
    grid  = NGrid(kgrid, tgrid)

    @testset "currentvalue" begin
        @test currentvalue(PowerLawTest(:dt, 0.5), tgrid) == 0.1
        @test currentvalue(PowerLawTest(:kxmax, 0.5), kgrid) == 10.0
        @test currentvalue(LinearTest(:t0, -1.0), tgrid) == -5.0

        @test currentvalue(PowerLawTest(:dt, 0.5), grid) == 0.1
        @test currentvalue(PowerLawTest(:kxmax, 0.5), grid) == 10.0
    end
end
