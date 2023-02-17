### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 608b82c0-adf2-11ed-0850-ebf580639ec8
# remove once merged with master
using Pkg; Pkg.activate("..");

# ╔═╡ a1c1ef76-64d9-4f7b-b6b8-b70e180fc0ff
using RxInfer, Distributions, PlutoUI, PyPlot, PGFPlotsX, LaTeXStrings, Random, LinearAlgebra

# ╔═╡ 7c5873ce-a0d5-46f2-9d59-188ffd09cb5b
begin
	using SpecialFunctions
	@rule Categorical(:p, Marginalisation) (m_out::Categorical, q_out::PointMass) = begin
	    @logscale -SpecialFunctions.logfactorial(length(probvec(q_out)))
	    return Dirichlet(probvec(q_out) .+ one(eltype(probvec(q_out))))
	end

	struct EnforceMarginalFunctionalDependency <: ReactiveMP.AbstractNodeFunctionalDependenciesPipeline
	    edge :: Symbol
	end
	
	function ReactiveMP.message_dependencies(::EnforceMarginalFunctionalDependency, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
	    return ReactiveMP.message_dependencies(ReactiveMP.DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
	end
	
	function ReactiveMP.marginal_dependencies(enforce::EnforceMarginalFunctionalDependency, nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
	    default = ReactiveMP.marginal_dependencies(ReactiveMP.DefaultFunctionalDependencies(), nodeinterfaces, nodelocalmarginals, varcluster, cindex, iindex)
	    index   = ReactiveMP.findnext(i -> name(i) === enforce.edge, nodeinterfaces, 1)
	    if index === iindex 
	        return default
	    end
	    vmarginal = ReactiveMP.getmarginal(ReactiveMP.connectedvar(nodeinterfaces[index]), IncludeAll())
	    loc = ReactiveMP.FactorNodeLocalMarginal(-1, index, enforce.edge)
	    ReactiveMP.setstream!(loc, vmarginal)
	    # Find insertion position (probably might be implemented more efficiently)
	    insertafter = sum(first(el) < iindex ? 1 : 0 for el in default; init = 0)
	    return ReactiveMP.TupleTools.insertafter(default, insertafter, (loc, ))
	end

	# function for using hard switching
function ReactiveMP.functional_dependencies(::EnforceMarginalFunctionalDependency, factornode::MixtureNode{N, F}, iindex::Int) where {N, F <: FullFactorisation}
    message_dependencies = if iindex === 1
        # output depends on:
        (factornode.inputs,)
    elseif iindex === 2
        # switch depends on:
        (factornode.out, factornode.inputs)
    elseif 2 < iindex <= N + 2
        # k'th input depends on:
        (factornode.out, )
    else
        error("Bad index in functional_dependencies for SwitchNode")
    end

    marginal_dependencies = if iindex === 1
        # output depends on:
        (factornode.switch,)
    elseif iindex == 2
        #  switch depends on
        ()
    elseif 2 < iindex <= N + 2
        # k'th input depends on:
        (factornode.switch,)
    else
        error("Bad index in function_dependencies for SwitchNode")
    end
    # println(marginal_dependencies)
    return message_dependencies, marginal_dependencies
end

# create an observable that is used to compute the switch with pipeline constraints
function ReactiveMP.get_messages_observable(factornode::MixtureNode{N, F, Nothing, ReactiveMP.FactorNodePipeline{P, EmptyPipelineStage}}, messages::Tuple{ReactiveMP.NodeInterface, NTuple{N, ReactiveMP.IndexedNodeInterface}}) where {N, F <: FullFactorisation, P <: EnforceMarginalFunctionalDependency}
    switchinterface  = messages[1]
    inputsinterfaces = messages[2]

    msgs_names = Val{(name(switchinterface), name(inputsinterfaces[1]))}
    msgs_observable =
    combineLatest((ReactiveMP.messagein(switchinterface), combineLatest(map((input) -> ReactiveMP.messagein(input), inputsinterfaces), PushNew())), PushNew()) |>
        map_to((ReactiveMP.messagein(switchinterface), ReactiveMP.ManyOf(map((input) -> ReactiveMP.messagein(input), inputsinterfaces))))
    return msgs_names, msgs_observable
end

# create an observable that is used to compute the output with pipeline constraints
function ReactiveMP.get_messages_observable(factornode::MixtureNode{N, F, Nothing, ReactiveMP.FactorNodePipeline{P, EmptyPipelineStage}}, messages::Tuple{NTuple{N, ReactiveMP.IndexedNodeInterface}}) where {N, F <: FullFactorisation, P <: EnforceMarginalFunctionalDependency}
    inputsinterfaces = messages[1]

    msgs_names = Val{(name(inputsinterfaces[1]), )}
    msgs_observable =
    combineLatest(map((input) -> ReactiveMP.messagein(input), inputsinterfaces), PushNew()) |>
        map_to((ReactiveMP.ManyOf(map((input) -> ReactiveMP.messagein(input), inputsinterfaces)),))
    return msgs_names, msgs_observable
end

# create an observable that is used to compute the input with pipeline constraints
function ReactiveMP.get_messages_observable(factornode::MixtureNode{N, F, Nothing, ReactiveMP.FactorNodePipeline{P, EmptyPipelineStage}}, messages::Tuple{ReactiveMP.NodeInterface}) where {N, F <: FullFactorisation, P <: EnforceMarginalFunctionalDependency}
    outputinterface = messages[1]

    msgs_names = Val{(name(outputinterface), )}
    msgs_observable = combineLatestUpdates((ReactiveMP.messagein(outputinterface), ), PushNew())
    return msgs_names, msgs_observable
end

function ReactiveMP.get_marginals_observable(factornode::MixtureNode{N, F, Nothing, ReactiveMP.FactorNodePipeline{P, EmptyPipelineStage}}, marginals::Tuple{ReactiveMP.NodeInterface}) where {N, F <: FullFactorisation, P <: EnforceMarginalFunctionalDependency}
    switchinterface = marginals[1]

    marginal_names       = Val{(name(switchinterface), )}
    marginals_observable = combineLatestUpdates((getmarginal(ReactiveMP.connectedvar(switchinterface), IncludeAll()), ), PushNew())

    return marginal_names, marginals_observable
end

	# import Distributions: Dirichlet
	# using SpecialFunctions: loggamma
	# function Dirichlet{T}(alpha::AbstractVector{T}; check_args::Bool=true) where T
 #        alpha0 = sum(alpha)
 #        lmnB = sum(loggamma, alpha) - loggamma(alpha0)
 #        Dirichlet{T,typeof(alpha),typeof(lmnB)}(alpha, alpha0, lmnB)
 #    end
	
end

# ╔═╡ d2910ba3-6f0c-4905-b10a-f32ad8239ab6
md"""
# Validation experiments: online Dirichlet process training
"""

# ╔═╡ 2bfcbece-2b63-443d-95d9-de7479ded607
md"""
### Generate data
"""

# ╔═╡ fa56df5b-e46b-4ed7-a9a7-7ae1d8697250
md"""number of samples: $(@bind nr_samples Slider(1:2_000; default=1500, show_value=true))"""

# ╔═╡ 1ee92290-71ac-41ce-9666-241478bc04cb
begin
	means = (rand(MersenneTwister(123), 8, 2).-0.5)*40
	means_vec = [means[k,:] for k in 1:8]
	dist = MixtureModel(
		map(m -> MvNormal(m, I) ,means_vec),
		Categorical([0.1, 0.3, 0.2, 0.05, 0.15, 0.1, 0.05, 0.05])
	)
end;

# ╔═╡ b9550341-a928-4274-b9da-40ac84d2a991
data = rand(MersenneTwister(123), dist, nr_samples);

# ╔═╡ d6fbfce8-c87b-4e0e-b11c-4d9fdce25b51
begin
	plt.figure()
	plt.scatter(data[1,:], data[2,:], alpha=0.1)
	plt.xlabel(L"y_1")
	plt.ylabel(L"y_2")
	plt.xlim(-20, 20)
	plt.ylim(-20, 20)
	plt.gcf()
end

# ╔═╡ 4eee819f-f099-4d4d-b72b-434be3077f99
md"""
### Model specification
"""

# ╔═╡ e9c3f636-0049-4b42-b0f9-cc42bee61360
@model function model_dirichlet_process()

    # specify experimental outcomes
    y = datavar(Vector{Float64})

	# updatable parameters
    α = datavar(Vector{Float64})
	μ_θ = datavar(Vector{Float64}, 8)
	Λ_θ = datavar(Matrix{Float64}, 8)

	# other variables
	θk = randomvar(8)
	
    # specify initial distribution over clusters
    π ~ Dirichlet(α)
		
	# prior over model selection variable
	z ~ Categorical(π) where { pipeline = EnforceMarginalFunctionalDependency(:out) }

	# specify prior models over θ
	for k in 1:8
		θk[k] ~ MvNormalMeanPrecision(μ_θ[k], Λ_θ[k])
	end

	tθ = tuple(θk...)

	# specify mixture distribution
	θ ~ Mixture(z, tθ) where { pipeline = EnforceMarginalFunctionalDependency(:switch) }

	# specify observation noise
	y ~ MvNormalMeanPrecision(θ, diagm(ones(2)))

    return y, θ, θk, z, π, α

end

# ╔═╡ 5fd5ee6b-3208-42fe-b8b2-14e95ffd08b5
md"""
### Constraint specification
"""

# ╔═╡ 07b1d74f-d203-475c-834e-ae83459d714a
@constraints function constraints_dirichlet_process()
    q(z) :: PointMass
end

# ╔═╡ 9bc23d63-8693-4eb8-ae90-29ddc4ec4997
md"""
### Probabilistic inference
"""

# ╔═╡ 16e37d4b-523f-4d1b-ab71-b54ba81364b1
@bind alpha Slider(-5:5; default=0, show_value=false)

# ╔═╡ 6293a919-a314-4063-a679-c70008b12b2f
md"""alpha = 1e$(alpha)"""

# ╔═╡ 5f6aaef7-1deb-4812-9888-fc52009bdc5f
base_measure = MvNormalMeanPrecision(zeros(2), 0.1*diagm(ones(2)));

# ╔═╡ 37897bf5-c864-456f-87c0-1251ad532010
begin
	function update_alpha_vector(α_prev)
		ind = findfirst( isapprox(10.0^alpha + 1e-10), α_prev)
		if isnothing(ind)
			α_new = α_prev
		elseif ind == 2
			α_new = α_prev
			α_new[ind] = 10.0^alpha
		elseif ind > 2 && α_prev[ind-1] != 10.0^alpha
			α_new = α_prev
			α_new[ind] = 10.0^alpha
		else
			α_new = α_prev
		end
		return α_new
	end
	function update_alpha(dist)
		return update_alpha_vector(probvec(dist))
	end
	function broadcast_mean_precision(dist)
		tmp = mean_precision.(dist)
		return first.(tmp), last.(tmp)
	end
end;

# ╔═╡ aec5408c-aab9-4fba-9bf2-0bace3c2c29f
autoupdates_dirichlet_process = @autoupdates begin
    α = update_alpha(q(π))
	μ_θ, Λ_θ = broadcast_mean_precision(q(θk))
end;

# ╔═╡ 5f3b9e1f-2ffc-403b-95df-3e48504399bc
 function run_dirichlet_process(data)
	 alpha_start = 10.0^alpha * [1.0, 0, 0, 0, 0, 0, 0, 0] + 1e-10*ones(8)
	 return rxinference(
		model         = model_dirichlet_process(),
		data          = (y = [data[:,k] for k=1:size(data,2)], ),
		constraints   = constraints_dirichlet_process(),
		autoupdates   = autoupdates_dirichlet_process,
		initmarginals = (π = Dirichlet(alpha_start; check_args=false), θk = base_measure),
		returnvars    = (:π, :θk),
		keephistory   = size(data,2),
		historyvars   = (z = KeepLast(), π = KeepLast(), θk = KeepLast()),
		autostart     = true,
		addons        = AddonLogScale()
	)
 end;

# ╔═╡ 1b2c3587-b4b6-4d5b-a54a-430fc478dcd4
md"""
### Results
"""

# ╔═╡ 2bfa1683-86c3-4b9d-b7e3-3890bb32c645
results_dirichlet_process = run_dirichlet_process(data)

# ╔═╡ 075cbfdd-698b-4a38-8ae3-557d39acb5d2
md"""sample index: $(@bind N Slider(1:nr_samples; default=nr_samples, show_value=true))"""

# ╔═╡ e2ed2836-5a4d-4762-ab59-d77277b47f39
begin
	plt.figure()
	plt.scatter(data[1,1:N], data[2,1:N], alpha=0.1)
	for k in findall(x -> x > 1, probvec(results_dirichlet_process.history[:π][N]))
		plt.scatter(mean(results_dirichlet_process.history[:θk][N][k])[1], mean(results_dirichlet_process.history[:θk][N][k])[2], marker="x")
	end
	plt.xlabel(L"y_1")
	plt.ylabel(L"y_2")
	plt.xlim(-20, 20)
	plt.ylim(-20, 20)
	plt.gcf()
end

# ╔═╡ Cell order:
# ╟─d2910ba3-6f0c-4905-b10a-f32ad8239ab6
# ╠═608b82c0-adf2-11ed-0850-ebf580639ec8
# ╠═a1c1ef76-64d9-4f7b-b6b8-b70e180fc0ff
# ╟─2bfcbece-2b63-443d-95d9-de7479ded607
# ╟─fa56df5b-e46b-4ed7-a9a7-7ae1d8697250
# ╟─1ee92290-71ac-41ce-9666-241478bc04cb
# ╠═b9550341-a928-4274-b9da-40ac84d2a991
# ╟─d6fbfce8-c87b-4e0e-b11c-4d9fdce25b51
# ╟─4eee819f-f099-4d4d-b72b-434be3077f99
# ╠═7c5873ce-a0d5-46f2-9d59-188ffd09cb5b
# ╠═e9c3f636-0049-4b42-b0f9-cc42bee61360
# ╟─5fd5ee6b-3208-42fe-b8b2-14e95ffd08b5
# ╠═07b1d74f-d203-475c-834e-ae83459d714a
# ╟─9bc23d63-8693-4eb8-ae90-29ddc4ec4997
# ╟─16e37d4b-523f-4d1b-ab71-b54ba81364b1
# ╟─6293a919-a314-4063-a679-c70008b12b2f
# ╠═5f6aaef7-1deb-4812-9888-fc52009bdc5f
# ╟─37897bf5-c864-456f-87c0-1251ad532010
# ╠═aec5408c-aab9-4fba-9bf2-0bace3c2c29f
# ╠═5f3b9e1f-2ffc-403b-95df-3e48504399bc
# ╟─1b2c3587-b4b6-4d5b-a54a-430fc478dcd4
# ╠═2bfa1683-86c3-4b9d-b7e3-3890bb32c645
# ╟─075cbfdd-698b-4a38-8ae3-557d39acb5d2
# ╟─e2ed2836-5a4d-4762-ab59-d77277b47f39
