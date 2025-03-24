var documenterSearchIndex = {"docs":
[{"location":"Reference.html","page":"Reference","title":"Reference","text":"CurrentModule = Damysos","category":"page"},{"location":"Reference.html#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"Reference.html#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"Reference.html","page":"Reference","title":"Reference","text":"","category":"page"},{"location":"Reference.html#Docstrings","page":"Reference","title":"Docstrings","text":"","category":"section"},{"location":"Reference.html","page":"Reference","title":"Reference","text":"Modules = [Damysos]","category":"page"},{"location":"Reference.html#Damysos.m_e","page":"Reference","title":"Damysos.m_e","text":"Rest mass of an electron as Quantity (Unitful.jl package) equal to 9.1093837139(28)e-31kg\n\n\n\n\n\n","category":"constant"},{"location":"Reference.html#Damysos.q_e","page":"Reference","title":"Damysos.q_e","text":"Elementary charge as Quantity (Unitful.jl package) equal to 1.602176634e-19C\n\n\n\n\n\n","category":"constant"},{"location":"Reference.html#Damysos.ħ","page":"Reference","title":"Damysos.ħ","text":"Reduced Planck constant as Quantity (Unitful.jl package) equal to 6.582119569...e-16 eV⋅s\n\n\n\n\n\n","category":"constant"},{"location":"Reference.html#Damysos.BilayerToy","page":"Reference","title":"Damysos.BilayerToy","text":"BilayerToy{T<:Real} <: GeneralTwoBand{T}\n\nA toy model of touching quadratic bands with a gap turning them quartic near the band edge.\n\nThe Hamiltonian reads \n\nhatH = fraczeta2(k_y^2-k_x^2)sigma_x - k_x k_y sigma_y + fracDelta2sigma_z\n\nsuch that vech=zeta2 (k_y^2-k_x^2) -zeta2 k_x k_y Delta2.  The dimensionful form (SI) would be\n\nhatH_SI = frachbar^22m^*(k_y^2-k_x^2)sigma_x - k_x k_y sigma_y+fracE_gap2sigma_z\n\nExamples\n\njulia> h = BilayerToy(0.2,1.0)\nBilayerToy:\n  Δ: 0.2\n  ζ: 1.0\n \n\n\nSee also\n\nGeneralTwoBand GappedDirac\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.CartesianKGrid1d","page":"Reference","title":"Damysos.CartesianKGrid1d","text":"CartesianKGrid1d{T}(dkx, kxmax[, ky]) <: CartesianKGrid{T}\n\nOne-dimensional equidistant samples in k-space in kx direction at ky.\n\nSee also\n\nSimulation, SymmetricTimeGrid, KGrid0d, CartesianKGrid2d\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.CartesianKGrid2d","page":"Reference","title":"Damysos.CartesianKGrid2d","text":"CartesianKGrid2d{T}(dkx, kxmax, dky, kymax) <: CartesianKGrid{T}\n\nTwo-dimensional equidistant cartesian grid in reciprocal (k-)space.\n\nSee also\n\nSimulation, SymmetricTimeGrid, CartesianKGrid1d, KGrid0d\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.ConvergenceTest","page":"Reference","title":"Damysos.ConvergenceTest","text":"ConvergenceTest(start, solver::DamysosSolver = LinearChunked(); kwargs...)\n\nA convergence test based on a Simulation and a ConvergenceTestMethod.\n\nArguments\n\nstart: the starting point can be a Simulation object or are path to a previous test\nsolver::DamysosSolver = LinearChunked(): the solver used for simulations,\n\nKeyword arguments\n\nmethod::ConvergenceTestMethod: specifies the convergence parmeter & iteration method\nresume::Bool: if true re-use the completedsims, otherwise start from scratch\natolgoal::Real: desired absolute tolerance\nrtolgoal::Real: desired relative tolerance\nmaxtime::Union{Real,Unitful.Time}: test guaranteed to stop after maxtime\nmaxiterations::Integer: test stops after maxiterations Simulations were performed\npath::String: path to save data of convergence test\naltpath: path to try inf start.datapath throws an error\n\nSee also\n\nLinearTest, PowerLawTest, Simulation\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.ConvergenceTestResult","page":"Reference","title":"Damysos.ConvergenceTestResult","text":"Result of a ConvergenceTest\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.ExtendKymaxTest","page":"Reference","title":"Damysos.ExtendKymaxTest","text":"ExtendKymaxTest(extendmethod::ConvergenceTestMethod)\n\nSpecialized for extending the integration region in ky-direction avoided re-calculation.\n\nSee also\n\nConvergenceTest, LinearTest PowerLawTest\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.GappedDirac","page":"Reference","title":"Damysos.GappedDirac","text":"GappedDirac{T<:Real} <: GeneralTwoBand{T}\n\nMassive Dirac Hamiltonian (two-band model).\n\nThe Hamiltonian reads \n\nhatH = k_xsigma_x + k_ysigma_y + msigma_z\n\nsuch that vech=k_xk_ym.\n\nExamples\n\njulia> h = GappedDirac(1.0)\nGappedDirac:\n  m: 1.0\n  vF: 1.0\n\n\nSee also\n\nGeneralTwoBand QuadraticToy BilayerToy\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.GaussianAPulse","page":"Reference","title":"Damysos.GaussianAPulse","text":"GaussianAPulse{T<:Real}\n\nRepresents spacially homogeneous, linearly polarized pulse with Gaussian envelope. \n\nMathematical form\n\nThe form of the vector potential is given by\n\nvecA(t) = vecA_0 cos(omega t + theta) e^-t^2  2sigma^2\n\nwhere vecA_0=A_0(cosvarphivece_x + sinvarphivece_y). \n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.GaussianEPulse","page":"Reference","title":"Damysos.GaussianEPulse","text":"GaussianEPulse{T<:Real}\n\nRepresents spacially homogeneous, linearly polarized pulse with Gaussian envelope. \n\nMathematical form\n\nThe form of the electric field is given by\n\nvecE(t) = vecE_0 sin(omega t+phi) e^-t^2  sigma^2\n\nwhere vecE_0=E_0(cosvarphivece_x + sinvarphivece_y). \n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.GeneralTwoBand","page":"Reference","title":"Damysos.GeneralTwoBand","text":"GeneralTwoBand{T} <: Hamiltonian{T}\n\nSupertype of all 2x2 Hamiltonians with all matrixelements via dispatch.\n\nIdea\n\nThe idea is that all Hamiltonians of the form\n\nhatH = vech(veck)cdotvecsigma\n\ncan be diagonalized analytically and hence most desired matrixelements such as velocities or dipoles can be expressed solely through \n\nvech(veck)=h_x(veck)h_y(veck)h_z(veck)\n\nand its derivatives with respect to k_mu. Any particular Hamiltonian deriving form GeneralTwoBand{T} must then only implement vech(veck) and its derivatives.\n\nSee also\n\nGappedDirac\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.KGrid0d","page":"Reference","title":"Damysos.KGrid0d","text":"KGrid0d{T}(kx,ky) <: CartesianKGrid{T}\n\nZero-dimensional grid, i.e. a single point in k-space.\n\nSee also\n\nSimulation, SymmetricTimeGrid, CartesianKGrid1d, CartesianKGrid2d\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.LinearCUDA","page":"Reference","title":"Damysos.LinearCUDA","text":"LinearCUDA\n\nRepresents an integration strategy for k-space via simple midpoint sum, where individual k points are computed concurrently on one or several CUDA GPU(s) via linear indexing.\n\nFields\n\nkchunksize::Int64: number of k-points in one concurrently executed chunk. \nalgorithm::GPUODEAlgorithm: algorithm for solving differential equations\nngpus::Int: #GPUs to use, automatically chooses all available GPUs if none given\nrtol::Union{Nothing, Real}: relative tolerance of solver (nothing for fixed-timestep)\natol::Union{Nothing, Real}: absolute tolerance of solver (nothing for fixed-timestep)\n\nSee also\n\nLinearChunked, SingleMode\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.LinearChunked","page":"Reference","title":"Damysos.LinearChunked","text":"LinearChunked\n\nRepresents an integration strategy for k-space via simple midpoint sum.\n\nFields\n\nkchunksize::T: number of k-points in one chunk. Every task/worker gets one chunk. \nalgorithm::SciMLBase.BasicEnsembleAlgorithm: algorithm for the EnsembleProblem.\nodesolver::SciMLBase.AbstractODEAlgorithm: ODE algorithm\n\nExamples\n\njulia> solver = LinearChunked(256,EnsembleThreads())\nLinearChunked:\n  - kchunksize: 256\n  - algorithm: EnsembleThreads()\n  - odesolver: Vern7{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}(OrdinaryDiffEqCore.trivial_limiter!, OrdinaryDiffEqCore.trivial_limiter!, static(false), true)\n  \n\nSee also\n\nLinearChunked, SingleMode\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.LinearTest","page":"Reference","title":"Damysos.LinearTest","text":"LinearTest{T<:Real}(parameter::Symbol,shift{T})\n\nA convergence method where parameter is changed by adding shift each iteration.\n\nSee also\n\nConvergenceTest, PowerLawTest ExtendKymaxTest\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.NGrid","page":"Reference","title":"Damysos.NGrid","text":"NGrid{T}(kgrid, tgrid)\n\nRepresents the discretzation of a Simulation in reciprocal (k-)space and time.\n\nFields\n\nkgrid::KGrid{T}: set of points in k-space used for integration of observables.\ntgrid::TimeGrid{T}: points in time-space to evaluate observables.\n\nSee also\n\nSimulation, SymmetricTimeGrid, CartesianKGrid1d, CartesianKGrid2d KGrid0d\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.Occupation","page":"Reference","title":"Damysos.Occupation","text":"Occupation{T<:Real} <: Observable{T}\n\nHolds time series data of the occupation computed from the density matrix.\n\nOnly the conduction band occupation rho_cc(t) is stored since Trrho(t)=1\n\nFields\n\ncbocc::Vector{T}: time-dependent conduction band occupation rho_cc(t)\n\nSee also\n\nVelocity\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.PowerLawTest","page":"Reference","title":"Damysos.PowerLawTest","text":"PowerLawTest{T<:Real}(parameter::Symbol,multiplier{T})\n\nA convergence method multiplying parameter by multiplier each iteration.\n\nSee also\n\nConvergenceTest, LinearTest ExtendKymaxTest\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.QuadraticToy","page":"Reference","title":"Damysos.QuadraticToy","text":"QuadraticToy{T<:Real} <: GeneralTwoBand{T}\n\nA toy model of quadratic bands with a gap.\n\nThe Hamiltonian reads \n\nhatH = fraczeta2(k_x^2sigma_x + k_y^2sigma_y) + fracDelta2sigma_z\n\nsuch that vech=zeta2 k_x^2zeta2 k_y^2 Delta2. The dimensionful  form (SI) would be\n\nhatH_SI = frachbar^22m^*(k_x^2sigma_x+k_y^2sigma_y)+fracE_gap2sigma_z\n\nExamples\n\njulia> h = QuadraticToy(0.2,1.0)\nQuadraticToy:\n  Δ: 0.2\n  ζ: 1.0\n \n\n\nSee also\n\nGeneralTwoBand GappedDirac\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.Simulation","page":"Reference","title":"Damysos.Simulation","text":"Simulation{T}(l, df, g, obs, us, d[, id])\n\nRepresents a simulation with all physical and numerical parameters specified.\n\nFields\n\nl::Liouvillian{T}: describes physical system via Liouville operator\ndf::DrivingField{T}: laser field driving the system\ng::NGrid{T}: time & reciprocal (k-) space discretization\nobs::Vector{Observable{T}}: physical observables to be computed\nus::UnitScaling{T}: time- and lengthscale linking dimensionless units to SI units\nid::String: identifier of the Simulation\ndimensions::UInt8: system can be 0d (single mode),1d or 2d\n\nSee also\n\nNGrid, TwoBandDephasingLiouvillian, UnitScaling, Velocity, Occupation, GaussianAPulse\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.SingleMode","page":"Reference","title":"Damysos.SingleMode","text":"SingleMode\n\nRepresents the solver for a single point in k-space.\n\nSee also\n\nLinearChunked, LinearCUDA\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.SymmetricTimeGrid","page":"Reference","title":"Damysos.SymmetricTimeGrid","text":"SymmetricTimeGrid{T}(dt, t0)\n\nTime discretzation of a Simulation spanning (-t0,t0) in steps of dt.\n\nFields\n\ndt::T: timestep in internal dimensionless units (see UnitScaling).\nt0::T: .\n\nSee also\n\nSimulation, SymmetricTimeGrid, CartesianKGrid1d, CartesianKGrid2d\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.TwoBandDephasingLiouvillian","page":"Reference","title":"Damysos.TwoBandDephasingLiouvillian","text":"TwoBandDephasingLiouvillian{T<:Real} <: Liouvillian{T}\n\nRepresents a system with a two-band Hamiltonian and T_2 dephasing and T_1 relaxation\n\nSee also\n\nGappedDirac, GeneralTwoBand\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.UnitScaling","page":"Reference","title":"Damysos.UnitScaling","text":"UnitScaling(timescale,lengthscale)\n\nRepresents a physical length- and time-scale used for non-dimensionalization of a system.\n\nExamples\n\njulia> us = UnitScaling(u\"1.0s\",u\"1.0m\")\nUnitScaling:\n timescale: 1.0e15 fs\n lengthscale: 1.0e9 nm\n\n\n\nFurther information\n\nInternally, the fields timescale & lengthscale of UnitScaling are saved in femtoseconds  and nanometers, but never used for numerical calculations. They are only needed to convert to dimensionful quantities again (Unitful package used, supports SI,cgs,... units) See here for more information on non-dimensionalization.\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.Velocity","page":"Reference","title":"Damysos.Velocity","text":"Velocity{T<:Real} <: Observable{T}\n\nHolds time series data of the physical velocity computed from the density matrix.\n\nThe velocity is computed via\n\nvecv(t) = Tr rho(t) fracpartial Hpartialveck\n            = rho_cc(t) vecv_cc(t)  + rho_cv(t) vecv_vc(t) \n            + rho_vv(t) vecv_vv(t)  + rho_vc(t) vecv_cv(t) \n\nFields\n\nvx::Vector{T}: total velocity in x-direction. \nvxintra::Vector{T}: rho_cc(t) v^x_cc(t)  + rho_vv(t) v^x_vv(t)\nvxinter::Vector{T}: rho_cv(t) v^x_vc(t)  + rho_vc(t) v^x_cv(t)\nvy::Vector{T}: total velocity in y-direction. \nvyintra::Vector{T}: rho_cc(t) v^y_cc(t)  + rho_vv(t) v^y_vv(t)\nvyinter::Vector{T}: rho_cv(t) v^y_vc(t)  + rho_vc(t) v^y_cv(t)\n\nSee also\n\nOccupation\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.VelocityX","page":"Reference","title":"Damysos.VelocityX","text":"VelocityX{T<:Real} <: Observable{T}\n\nHolds time series data of the physical velocity in x direction.\n\nThe velocity is computed via\n\nvecv(t) = Tr rho(t) fracpartial Hpartialveck\n            = rho_cc(t) vecv_cc(t)  + rho_cv(t) vecv_vc(t) \n            + rho_vv(t) vecv_vv(t)  + rho_vc(t) vecv_cv(t) \n\nFields\n\nvx::Vector{T}: total velocity in x-direction. \nvxintra::Vector{T}: rho_cc(t) v^x_cc(t)  + rho_vv(t) v^x_vv(t)\nvxinter::Vector{T}: rho_cv(t) v^x_vc(t)  + rho_vc(t) v^x_cv(t)\n\nSee also\n\nOccupation Velocity\n\n\n\n\n\n","category":"type"},{"location":"Reference.html#Damysos.define_functions","page":"Reference","title":"Damysos.define_functions","text":"define_functions(sim,solver)\n\nHardcode the functions needed to run the Simulation. \n\nArguments\n\nsim::Simulation: contains physical & numerical information (see Simulation)\nsolver: strategy for integrating in k-space. Defaults to LinearChunked)\n\nReturns\n\nVector of functions used by run!.\n\nSee also\n\nSimulation, run!, LinearChunked\n\n\n\n\n\n","category":"function"},{"location":"Reference.html#Damysos.dx_cc-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.dx_cc","text":"dx_cc(h,kx,ky)\n\nReturns the dipole operator matrix element ⟨ψ+|x|ψ+⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); dx_cc(h,1.0,-1.0)\n-0.10566243270259357\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.dx_cv-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.dx_cv","text":"dx_cv(h,kx,ky)\n\nReturns the dipole operator matrix element ⟨ψ+|x|ψ-⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); dx_cv(h,1.0,-1.0)\n-0.0610042339640731 - 0.22767090063073978im\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.dx_vc-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.dx_vc","text":"dx_vc(h,kx,ky)\n\nReturns the dipole operator matrix element ⟨ψ-|x|ψ+⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); dx_vc(h,1.0,-1.0)\n-0.0610042339640731 + 0.22767090063073978im\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.dx_vv-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.dx_vv","text":"dx_vv(h,kx,ky)\n\nReturns the dipole operator matrix element ⟨ψ-|x|ψ-⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); dx_vv(h,1.0,-1.0)\n0.10566243270259357\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.dy_cc-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.dy_cc","text":"dy_cc(h,kx,ky)\n\nReturns the dipole operator matrix element ⟨ψ+|y|ψ+⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); dy_cc(h,1.0,-1.0)\n-0.10566243270259357\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.dy_cv-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.dy_cv","text":"dy_cv(h,kx,ky)\n\nReturns the dipole operator matrix element ⟨ψ+|y|ψ-⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); dy_cv(h,1.0,-1.0)\n-0.22767090063073978 - 0.0610042339640731im\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.dy_vc-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.dy_vc","text":"dy_vc(h,kx,ky)\n\nReturns the dipole operator matrix element ⟨ψ-|y|ψ+⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); dy_vc(h,1.0,-1.0)\n-0.22767090063073978 + 0.0610042339640731im\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.dy_vv-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.dy_vv","text":"dy_vv(h,kx,ky)\n\nReturns the dipole operator matrix element ⟨ψ-|y|ψ-⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); dy_vv(h,1.0,-1.0)\n0.10566243270259357\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.estimate_atol-Tuple{GeneralTwoBand}","page":"Reference","title":"Damysos.estimate_atol","text":"Estimate the largest numeric scale in a Hamiltonian\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.hmat-Union{Tuple{T}, Tuple{GeneralTwoBand{T}, Any, Any}} where T","page":"Reference","title":"Damysos.hmat","text":"hmat(h,kx,ky)\n\nReturns matrixelements of Hamiltonian h at veck=k_xk_y (in diabatic basis).\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.hvec-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.hvec","text":"hvec(h,kx,ky)\n\nReturns vech(veck) defining the Hamiltonian at veck=k_xk_y.\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.linear_fit-Tuple{Any, Any}","page":"Reference","title":"Damysos.linear_fit","text":"Fits a straight line through a set of points and returns an anonymous fit-function\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.run!","page":"Reference","title":"Damysos.run!","text":"run!(sim, functions[, solver]; kwargs...)\n\nRun a simulation.\n\nArguments\n\nsim::Simulation: contains physical & numerical information (see Simulation)\nfunctions: needed by the solver/integrator (see define_functions).\nsolver: strategy for integrating in k-space. Defaults to LinearChunked)\n\nKeyword Arguments\n\nsavedata::Bool: save observables and simulation to disk after completion\nplotdata::Bool: create default plots and save them to disk after completion\nsavepath::String: path to directory to save data & plots\nshowinfo::Bool: log/display simulation info before running\nnan_limit::Int: maximum tolerated number of nans in observables\n\nReturns\n\nThe observables obtained from the simulation.\n\nSee also\n\nSimulation, define_functions, LinearChunked\n\n\n\n\n\n","category":"function"},{"location":"Reference.html#Damysos.run!-Tuple{ConvergenceTest}","page":"Reference","title":"Damysos.run!","text":"run!(test::ConvergenceTest; kwargs...)\n\nRun a convergence test and return the result as ConvergenceTestResult.\n\nKeyword Arguments\n\nsavedata::Bool: save observables and simulations to disk\nfilepath::String: path to .hdf5 file to save results\nnan_limit::Int: maximum tolerated number of nans in observables\nmax_nan_retries::Int: maximum number of iterations where nan_limit nans are tolerated\n\nReturns\n\nA ConvergenceTestResult object.\n\nSee also\n\nConvergenceTest\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.successful_retcode-Tuple{ConvergenceTestResult}","page":"Reference","title":"Damysos.successful_retcode","text":"successful_retcode(x)\n\nReturns true if x terminated with a successful return code.\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.successful_retcode-Tuple{String}","page":"Reference","title":"Damysos.successful_retcode","text":"successful_retcode(path::String)\n\nLoads .hdf5 file of convergence test and returns true if it was successful.\n\nSee also\n\nConvergenceTest\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.terminated_retcode-Tuple{Damysos.ReturnCode.T}","page":"Reference","title":"Damysos.terminated_retcode","text":"terminated_retcode(x)\n\nReturns true if x terminated regularly.\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.terminated_retcode-Tuple{String}","page":"Reference","title":"Damysos.terminated_retcode","text":"terminated_retcode(path::String)\n\nLoads .hdf5 file of convergence test and returns true if it terminated regularly.\n\nSee also\n\nConvergenceTest\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.vx_cc-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.vx_cc","text":"vx_cc(h,kx,ky)\n\nReturns the velocity operator matrix element ⟨ψ+|vx|ψ+⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); vx_cc(h,1.0,-1.0)\n0.5773502691896258\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.vx_cv-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.vx_cv","text":"vx_cv(h,kx,ky)\n\nReturns the velocity operator matrix element ⟨ψ+|vx|ψ-⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); vx_cv(h,1.0,-1.0)\n0.7886751345948129 - 0.21132486540518708im\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.vx_vc-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.vx_vc","text":"vx_vc(h,kx,ky)\n\nReturns the velocity operator matrix element ⟨ψ-|vx|ψ+⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); vx_vc(h,1.0,-1.0)\n0.7886751345948129 + 0.21132486540518708im\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.vx_vv-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.vx_vv","text":"vx_vv(h,kx,ky)\n\nReturns the velocity operator matrix element ⟨ψ-|vx|ψ-⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); vx_vv(h,1.0,-1.0)\n-0.5773502691896258\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.vy_cc-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.vy_cc","text":"vy_cc(h,kx,ky)\n\nReturns the velocity operator matrix element ⟨ψ+|vy|ψ+⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); vy_cc(h,1.0,-1.0)\n-0.5773502691896258\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.vy_cv-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.vy_cv","text":"vy_cv(h,kx,ky)\n\nReturns the velocity operator matrix element ⟨ψ+|vy|ψ-⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); vy_cv(h,1.0,-1.0)\n0.21132486540518708 - 0.7886751345948129im\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.vy_vc-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.vy_vc","text":"vy_vc(h,kx,ky)\n\nReturns the velocity operator matrix element ⟨ψ-|vy|ψ+⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); vy_vc(h,1.0,-1.0)\n0.21132486540518708 + 0.7886751345948129im\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.vy_vv-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.vy_vv","text":"vy_vv(h,kx,ky)\n\nReturns the velocity operator matrix element ⟨ψ-|vy|ψ-⟩ at veck=k_xk_y.\n\nExample\n\njulia> h = GappedDirac(1.0); vy_vv(h,1.0,-1.0)\n0.5773502691896258\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.Δϵ-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.Δϵ","text":"Δϵ(h,kx,ky)\n\nReturns the band energy (valence & conduction) difference at veck=k_xk_y.\n\n\n\n\n\n","category":"method"},{"location":"Reference.html#Damysos.ϵ-Tuple{GeneralTwoBand, Any, Any}","page":"Reference","title":"Damysos.ϵ","text":"ϵ(h,kx,ky)\n\nReturns the eigenenergy of the positive (conduction band) state at veck=k_xk_y.\n\n\n\n\n\n","category":"method"},{"location":"index.html","page":"Home","title":"Home","text":"CurrentModule = Damysos","category":"page"},{"location":"index.html#Damysos","page":"Home","title":"Damysos","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"A package to solve the Semiconductor Bloch equations using GPU or CPU.","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Documentation for Damysos.","category":"page"},{"location":"index.html#Package-features","page":"Home","title":"Package features","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Modular building blocks (driving field, Hamiltonian etc. ) for creating simulations\nUsing velocity-gauge to parallelize over k-points\nRun any Simulation on GPU or CPU without re-writing any code\nBuilt using the powerful DifferentialEquations.jl and DiffEqGPU.jl","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"","category":"page"}]
}
