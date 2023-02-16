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

# ╔═╡ 34771c7d-de6a-44c8-aa78-4e991f62f237
# remove once merged with master
using Pkg; Pkg.activate("..");

# ╔═╡ cd9431ce-ac3c-11ed-02c9-5b5600e798bb
using RxInfer, Distributions, Random, PyPlot, PlutoUI, SpecialFunctions, LinearAlgebra, PGFPlotsX

# ╔═╡ 5129f69a-43da-4659-9eb4-8d0723eea615
html"""<style>
main {
    max-width: 800px;
}
"""

# ╔═╡ e8f2c621-c640-4683-a8bd-d82a9f7dd73f
md"""
# Verification experiments
"""

# ╔═╡ ed0dbb47-764e-4dc6-8d58-5623e582692c
begin
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}");
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{bm}");
end;

# ╔═╡ efd8d329-d6be-4a47-884e-776d7e5f7a43
md"""
Number of samples
"""

# ╔═╡ 0a8bc1f9-7313-4027-b841-f778b31c91d8
@bind nr_samples Slider(1:2_000; default=1500, show_value=true)

# ╔═╡ 34ab39fd-02c3-4e12-857b-3673315cf799
dist = MixtureModel(Normal, [(-4.0, 1.0), (0.0, 1.0), (5.0, 1.0)], [0.2, 0.5, 0.3]);

# ╔═╡ b9949a48-89d3-4f31-8d48-375ef6c22f84
noise_var = 1e-1

# ╔═╡ 01060a9a-4cba-4543-924b-2ef9927be903
function generate_data(dist, nr_samples; rng=MersenneTwister(123))

    # sample from distribution
    samples = rand(rng, dist, nr_samples)

    # return samples
    return samples + rand(rng, Normal(0, sqrt(noise_var)), nr_samples)

end;

# ╔═╡ 51a5308a-65f7-42d2-90f8-f9fb1de41847
data = generate_data(dist, nr_samples);

# ╔═╡ cca8814e-600d-4a65-8444-5f7fcfdb5fc8
begin
	plt.figure()
	plt.plot(-8:0.01:8, map(x -> pdf(dist, x), -8:0.01:8), color="blue", linewidth=2)
	plt.hist(data, density="relative", bins=Int(round(10*log(nr_samples+1))), alpha=0.5)
	plt.plot(-8:0.01:8, map(x -> dist.prior.p[1]*pdf(dist.components[1], x), -8:0.01:8), color="red", linestyle="--")
	plt.plot(-8:0.01:8, map(x -> dist.prior.p[2]*pdf(dist.components[2], x), -8:0.01:8), color="red", linestyle="--")
	plt.plot(-8:0.01:8, map(x -> dist.prior.p[3]*pdf(dist.components[3], x), -8:0.01:8), color="red", linestyle="--")
	plt.grid()
	plt.xlim(-8, 8)
	plt.ylim(0, 0.25)
	plt.xlabel(L"y")
	plt.ylabel(L"p(y)")
	plt.gcf()
end

# ╔═╡ dbbaea7b-4a89-46a4-938c-b1ba1d8a7f7a
md"""
### Model averaging
"""

# ╔═╡ 6666b133-8544-4a71-bc3d-ab59c1bc59ea
@model function model_averaging(nr_samples)

    # specify experimental outcomes
    y = datavar(Float64, nr_samples)
    θ = randomvar(nr_samples)
    θ1 = randomvar(nr_samples)
    θ2 = randomvar(nr_samples)
    θ3 = randomvar(nr_samples)

    # prior over model selection variable
    z ~ Categorical([1/3, 1/3, 1/3])

    # create likelihood models
    for i in 1:nr_samples

        # specify prior models over θ
        θ1[i] ~ NormalMeanPrecision(-4, 1)
        θ2[i] ~ NormalMeanPrecision(0, 1)
        θ3[i] ~ NormalMeanPrecision(5, 1)

        # specify mixture distribution
        θ[i] ~ Mixture(z, (θ1[i], θ2[i], θ3[i]))

        # specify observation noise
        y[i] ~ NormalMeanPrecision(θ[i], 1/noise_var)
        
    end

    return y, θ, θ1, θ2, θ3, z

end

# ╔═╡ bf7f3200-1189-4cfe-90c1-3091add478a4
function run_averaging(data)
	return inference(
	    model = model_averaging(length(data)), 
	    data  = (y = data, ),
	    returnvars = (θ = KeepLast(), θ1 = KeepLast(), θ2 = KeepLast(), θ3 = KeepLast(), 					z=KeepLast()),
	    addons = AddonLogScale()
	)
end

# ╔═╡ 9a002685-5228-4245-a834-385c1fe10321
results_averaging = run_averaging(data)

# ╔═╡ b81effdb-e88a-471e-bf64-655e4fb3f2c4
begin
	plt.figure()
	plt.bar(1:length(probvec(results_averaging.posteriors[:z])), probvec(results_averaging.posteriors[:z]))
	plt.xlabel(L"k")
	plt.ylabel(L"p(z=k\mid y_{1:N})")
	plt.xticks(1:length(probvec(results_averaging.posteriors[:z])),1:length(probvec(results_averaging.posteriors[:z])) )
	plt.grid()
	plt.gcf()
end

# ╔═╡ 0dd9de2b-4047-4cf8-8626-78099d39cb6d
md"""
### Model selection
"""

# ╔═╡ 2a3928e8-c79c-4471-b2b1-4bc6fcf7dd52
@model function model_selection(nr_samples)

    # specify experimental outcomes
    y = datavar(Float64, nr_samples)
    θ = randomvar(nr_samples)
    θ1 = randomvar(nr_samples)
    θ2 = randomvar(nr_samples)
    θ3 = randomvar(nr_samples)

    # prior over model selection variable
    z ~ Categorical([1/3, 1/3, 1/3])

    # create likelihood models
    for i in 1:nr_samples

        # specify prior models over θ
        θ1[i] ~ NormalMeanPrecision(-4, 1)
        θ2[i] ~ NormalMeanPrecision(0, 1)
        θ3[i] ~ NormalMeanPrecision(5, 1)

        # specify mixture distribution
        θ[i] ~ Mixture(z, (θ1[i], θ2[i], θ3[i])) where { pipeline = RequireMarginal(switch) }

        # specify observation noise
        y[i] ~ NormalMeanPrecision(θ[i], 1/noise_var)
        
    end

    return y, θ, θ1, θ2, θ3, z

end

# ╔═╡ 584beb0f-b6ae-4323-85e8-15657e82d36a
@constraints function constraints_selection()
    q(z) :: PointMass
end

# ╔═╡ 2d83927a-32c0-402e-9f7a-151daef491a5
function run_selection(data)
	return inference(
	    model = model_selection(length(data)),
	    constraints = constraints_selection(),
	    data  = (y = data, ),
	    returnvars = (θ = KeepLast(), θ1 = KeepLast(), θ2 = KeepLast(), θ3 = KeepLast(), 					z=KeepLast()),
	    addons = AddonLogScale()
	)
end

# ╔═╡ 00f83269-9d3b-47b7-87b6-8aa1d8e8cc06
results_selection = run_selection(data)

# ╔═╡ bc727e09-6b41-4c6c-906d-5cf94263e164
begin
	plt.figure()
	plt.bar(1:length(mean(results_selection.posteriors[:z])), mean(results_selection.posteriors[:z]))
	plt.xlabel(L"k")
	plt.ylabel(L"p(z=k\mid y_{1:N})")
	plt.xticks(1:length(mean(results_selection.posteriors[:z])),1:length(mean(results_selection.posteriors[:z])) )
	plt.grid()
	plt.gcf()
end

# ╔═╡ eaefe9da-74ae-461b-bcd6-60d00b8a8c46
md"""
### Model combination (online)
"""

# ╔═╡ 94316a76-e202-4170-96a1-b6d346aaa806
begin
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
	
end

# ╔═╡ 0326ea66-0467-4462-872a-7d701fd14d01
@model function model_combination()

    # specify experimental outcomes
    y = datavar(Float64)
    α = datavar(Vector{Float64})
	
    # specify initial distribution over clusters
    π ~ Dirichlet(α)
		
	# prior over model selection variable
	z ~ Categorical(π) where { pipeline = EnforceMarginalFunctionalDependency(:out) }

	# specify prior models over θ
	θ1 ~ NormalMeanPrecision(-4, 1)
	θ2 ~ NormalMeanPrecision(0, 1)
	θ3 ~ NormalMeanPrecision(5, 1)

	# specify mixture distribution
	θ ~ Mixture(z, (θ1, θ2, θ3)) where { pipeline = EnforceMarginalFunctionalDependency(:switch) }

	# specify observation noise
	y ~ NormalMeanPrecision(θ, 1/noise_var)

    return y, θ, θ1, θ2, θ3, z, α

end

# ╔═╡ 1b7af4cb-27a6-4f68-aca5-544d3385efab
@constraints function constraints_combination()
    q(z) :: PointMass
end

# ╔═╡ 050018da-382a-45d7-a251-6c4dfb3c1c40
autoupdates_combination = @autoupdates begin 
    α = probvec(q(π))
end

# ╔═╡ 222663b1-03a8-4761-aaf1-23b476b037b4
 function run_combination(data)
	 return rxinference(
		model         = model_combination(),
		data          = (y = data, ),
		constraints   = constraints_combination(),
		autoupdates   = autoupdates_combination,
		initmarginals = (π = Dirichlet(ones(3)./3*length(data)), ),
		returnvars    = (:π, ),
		keephistory   = length(data),
		historyvars   = (z = KeepLast(), π = KeepLast()),
		autostart     = true,
		addons        = AddonLogScale()
	)
 end

# ╔═╡ af4b9dc8-38d4-48d1-b964-dab9fe6de033
results_combination = run_combination(data)

# ╔═╡ af34a2c1-d45e-4245-89fb-e6993fcb968f
begin
	plt.figure()
	plt.bar(1:length(probvec(results_combination.history[:π][end])), normalize(probvec(results_combination.history[:π][end]) - ones(3)./3*nr_samples, 1))
	plt.xlabel(L"k")
	plt.ylabel(L"p(z=k\mid y_{1:N})")
	plt.xticks(1:length(probvec(results_combination.history[:π][end])),1:length(probvec(results_combination.history[:π][end])) )
	plt.grid()
	plt.gcf()
end

# ╔═╡ cf49c906-e2a6-48a4-9fbe-c90aa18bf926
md"""
### Model combination (variational)
"""

# ╔═╡ 4fc49a89-863a-4ad6-b851-df30d76aa1df
@model function model_combination_variational(nr_samples)

    # specify experimental outcomes
    y = datavar(Float64, nr_samples)
	z = randomvar(nr_samples)
	θ1 = randomvar(nr_samples)
	θ2 = randomvar(nr_samples)
	θ3 = randomvar(nr_samples)
	θ = randomvar(nr_samples)
	
    # specify initial distribution over clusters
    π ~ Dirichlet(ones(3) ./ 3)

	for i in 1:nr_samples
		
		# prior over model selection variable
		z[i] ~ Categorical(π)
	
		# specify prior models over θ
		θ1[i] ~ NormalMeanPrecision(-4, 1)
		θ2[i] ~ NormalMeanPrecision(0, 1)
		θ3[i] ~ NormalMeanPrecision(5, 1)
	
		# specify mixture distribution
		θ[i] ~ Mixture(z[i], (θ1[i], θ2[i], θ3[i]))
	
		# specify observation noise
		y[i] ~ NormalMeanPrecision(θ[i], 1/noise_var)

	end

    return y, θ, θ1, θ2, θ3, z, π

end

# ╔═╡ eb881756-9982-4e78-81e6-2057a5e1c769
@constraints function constraints_combination_variational()
    q(π, z) = q(π)q(z)
end

# ╔═╡ 9434a98e-fef7-4f45-8cfb-5a82e0a1d587
begin
	tmp = 0
	@rule Categorical(:p, Marginalisation) (q_out::Any,) = begin
		@logscale 0
    	return Dirichlet(probvec(q_out) .+ one(eltype(probvec(q_out))))
	end
	@rule Categorical(:out, Marginalisation) (q_p::Dirichlet,) = begin
		@logscale 0
	    rho = clamp.(exp.(mean(log, q_p)), tiny, Inf) # Softens the parameter
	    return Categorical(rho ./ sum(rho))
	end
	function ReactiveMP.message_mapping_addon(::AddonLogScale{Nothing}, mapping, messages, marginals, result::Distribution)
	    # Here we assume
	    # 1. If log-scale value has not been computed during the message update rule
	    # 2. Either all messages or marginals are of type PointMass
	    # 3. The result of the message update rule is a proper distribution
	    #  THEN: logscale is equal to zero
	    #  OTHERWISE: show an error
	    #  This logic probably can be improved, e.g. if some tracks conjugacy between the node and messages
	    if isnothing(marginals) && all(data -> data isa PointMass, messages)
	        return AddonLogScale(0)
	    elseif isnothing(messages) && all(data -> data isa PointMass, marginals)
	        return AddonLogScale(0)
	    else
			return AddonLogScale(0)
	        # error("Log-scale value has not been computed for the message update rule = $(mapping)")
	    end
	end
end

# ╔═╡ cd69f55b-42c1-418e-956c-8da91b629672
 function run_combination_variational(data)
	 tmp
	 return inference(
		 model         = model_combination_variational(length(data)),
		 data          = (y = data, ),
		 constraints   = constraints_combination_variational(),
		 initmarginals = (π = vague(Dirichlet, 3), z = vague(Categorical, 3)),
		 returnvars    = (π=KeepLast(), z=KeepLast()),
		 addons        = AddonLogScale(),
		 iterations    = 10
	)
 end

# ╔═╡ f1d7aa95-5c64-43ac-8cd9-b131e75f05da
results_combination_variational = run_combination_variational(data)

# ╔═╡ 876c9a51-4137-4720-94f6-5192403ea217
begin
	plt.figure()
	plt.bar(1:length(probvec(results_combination_variational.posteriors[:π])), mean(results_combination_variational.posteriors[:π]), yerr=sqrt.(var(results_combination_variational.posteriors[:π])))
	plt.xlabel(L"k")
	plt.ylabel(L"p(z=k\mid y_{1:N})")
	plt.xticks(1:length(probvec(results_combination_variational.posteriors[:π])),1:length(probvec(results_combination_variational.posteriors[:π])) )
	plt.grid()
	plt.gcf()
end

# ╔═╡ 59c7a4e3-54f6-4da9-9257-807578516b09
begin
	plt.figure(figsize=(15,5))
	plt.scatter(data, zeros(nr_samples).+rand(nr_samples), c=argmax.(probvec.(results_combination_variational.posteriors[:z])))
	plt.xlim(-6,6)
	plt.xlabel(L"y")
	plt.grid()
	plt.gcf()
end

# ╔═╡ 4c891139-3175-4408-9bb8-571aaaa958c0
md"""
### Overview
"""

# ╔═╡ 66ad030d-3ccd-438f-b126-c3324707c53e
begin
	
	plt.figure(figsize=(15,25))

	# first row
	ax = plt.subplot(6, 4, (1,4))
	ax.plot(-6:0.01:6, map(x -> pdf(dist, x), -6:0.01:6), color="blue", linewidth=2)
	ax.plot(-6:0.01:6, map(x -> dist.prior.p[1]*pdf(dist.components[1], x), -6:0.01:6), color="red", linestyle="--")
	ax.plot(-6:0.01:6, map(x -> dist.prior.p[2]*pdf(dist.components[2], x), -6:0.01:6), color="red", linestyle="--")
	ax.plot(-6:0.01:6, map(x -> dist.prior.p[3]*pdf(dist.components[3], x), -6:0.01:6), color="red", linestyle="--")
	ax.grid()
	ax.set_xlabel(L"y")
	ax.set_ylabel(L"p(y)")
	ax.set_xlim(-6, 6)
	ax.set_ylim(0, 0.25)

	# rows
	for (i, k) in enumerate([1, 5, 10, 100, 1000])
		
		datax = generate_data(dist, k);

		# model averaging
		axx = plt.subplot(6, 4, 1+4*i)
		i == 1 ? axx.set_title("Model averaging") : nothing
		av = run_averaging(datax)
		axx.bar(1:length(probvec(av.posteriors[:z])), probvec(av.posteriors[:z]))
		axx.set_xlabel(L"k")
		axx.set_ylabel(L"p(z=k\mid y_{1:N})")
		axx.set_xticks(1:length(probvec(av.posteriors[:z])),1:length(probvec(av.posteriors[:z])) )
		axx.grid()

		# model selection
		axx = plt.subplot(6, 4, 2+4*i)
		i == 1 ? axx.set_title("Model selection") : nothing
		se = run_selection(datax)
		axx.bar(1:length(mean(se.posteriors[:z])), mean(se.posteriors[:z]))
		axx.set_xlabel(L"k")
		axx.set_ylabel(L"p(z=k\mid y_{1:N})")
		axx.set_xticks(1:length(mean(se.posteriors[:z])),1:length(mean(se.posteriors[:z])) )
		axx.grid()

		# model combination (online)
		axx = plt.subplot(6, 4, 3+4*i)
		i == 1 ? axx.set_title("Model combination (online)") : nothing
		co = run_combination(datax)
		axx.bar(1:length(probvec(co.history[:π][end])), normalize(probvec(co.history[:π][end]) - ones(3)./3*(k-1), 1))
		axx.set_xlabel(L"k")
		axx.set_ylabel(L"\mathrm{E}[z]")
		axx.set_xticks(1:length(probvec(co.history[:π][end])), 1:length(probvec(co.history[:π][end])))
		axx.grid()

		# model combination (variational)
		axx = plt.subplot(6, 4, 4+4*i)
		i == 1 ? axx.set_title("Model combination (variational)") : nothing
		co2 = run_combination_variational(datax)
		axx.bar(1:length(probvec(co2.posteriors[:π])), probvec(co2.posteriors[:π]), yerr=sqrt.(var(co2.posteriors[:π])))
		axx.set_xlabel(L"k")
		axx.set_ylabel(L"p(z=k\mid y_{1:N})")
		axx.set_xticks(1:length(probvec(co2.posteriors[:π])), 1:length(probvec(co2.posteriors[:π])))
		axx.grid()
		
	end
	plt.tight_layout()
	plt.gcf()
end

# ╔═╡ dda9a9e3-6e59-4671-adb3-c7b90144c46d
begin

# plot settings
bar_width = "20pt"

# prepare data
data1 = generate_data(dist, 1)
av1 = run_averaging(data1)
se1 = run_selection(data1)
cop1 = run_combination(data1)
cov1 = run_combination_variational(data1)
data5 = generate_data(dist, 5)
av5 = run_averaging(data5)
se5 = run_selection(data5)
cop5 = run_combination(data5)
cov5 = run_combination_variational(data5)
data10 = generate_data(dist, 10)
av10 = run_averaging(data10)
se10 = run_selection(data10)
cop10 = run_combination(data10)
cov10 = run_combination_variational(data10)
data100 = generate_data(dist, 100)
av100 = run_averaging(data100)
se100 = run_selection(data100)
cop100 = run_combination(data100)
cov100 = run_combination_variational(data100)
data1000 = generate_data(dist, 1000)
av1000 = run_averaging(data1000)
se1000 = run_selection(data1000)
cop1000 = run_combination(data1000)
cov1000 = run_combination_variational(data1000)
	
# create plot
fig_tikz = @pgf GroupPlot(

	# group plot options
	{
		group_style = {
			group_size = "4 by 5",
			horizontal_sep = "1cm",
		},
		label_style={font="\\footnotesize"},
		ticklabel_style={font="\\scriptsize",},
		xtick = [1, 2, 3],
        grid = "major",
		ymin = 0,
		width = "2in",
		height = "2in",
		ylabel_shift = "-5pt",
		xlabel_shift = "-5pt",
		xlabel = "\$k\$",
	},

	# axis row 1, column 1
	{
		ybar,
		bar_width=bar_width,
		ylabel_style={align="center"},
		ylabel = "\$\\bm{N=1}\$ \\\\ \\\\ \$p(z=k \\,\\vert\\, y_{1:N})\$",
        style = {thick},
		title = "\\textbf{Model averaging}\\\\",
		title_style={align="center"}
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(probvec(av1.posteriors[:z])), probvec(av1.posteriors[:z]))
    ),

	# axis row 1, column 2
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$q(z=k)\$",
        style = {thick},
		title = "\\textbf{Model selection}\\\\",
		title_style={align="center"}
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(se1.posteriors[:z])), mean(se1.posteriors[:z]))
    ),

	# axis row 1, column 3
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$\\mathbb{E}_{q(\\pi)}[\\pi_k]\$",
        style = {thick},
		title = "\\textbf{Model combination}\\\\(online)",
		title_style={align="center", yshift="-3pt"},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(cop1.history[:π][end])), normalize(probvec(cop1.history[:π][end]) - (1-1)/3*ones(3), 1))
    ),

	# axis row 1, column 4
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$\\mathbb{E}_{q(\\pi)}[\\pi_k]\$",
        style = {thick},
		title = "\\textbf{Model combination}\\\\(variational)",
		title_style={align="center", yshift="-3pt"},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(cov1.posteriors[:π])), mean(cov1.posteriors[:π]))
    ),

	# axis row 2, column 1
	{
		ybar,
		bar_width=bar_width,
		ylabel_style={align="center"},
		ylabel = "\$\\bm{N=5}\$ \\\\ \\\\ \$p(z=k \\,\\vert\\, y_{1:N})\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(probvec(av5.posteriors[:z])), probvec(av5.posteriors[:z]))
    ),

	# axis row 2, column 2
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$q(z=k)\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(se5.posteriors[:z])), mean(se5.posteriors[:z]))
    ),

	# axis row 2, column 3
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$\\mathbb{E}_{q(\\pi)}[\\pi_k]\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(cop5.history[:π][end])), normalize(probvec(cop5.history[:π][end]) - (5-1)/3*ones(3), 1))
    ),

	# axis row 2, column 4
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$\\mathbb{E}_{q(\\pi)}[\\pi_k]\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(cov5.posteriors[:π])), mean(cov5.posteriors[:π]))
    ),

	# axis row 3, column 1
	{
		ybar,
		bar_width=bar_width,
		ylabel_style={align="center"},
		ylabel = "\$\\bm{N=10}\$ \\\\ \\\\ \$p(z=k \\,\\vert\\, y_{1:N})\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(probvec(av10.posteriors[:z])), probvec(av10.posteriors[:z]))
    ),

	# axis row 3, column 2
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$q(z=k)\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(se10.posteriors[:z])), mean(se10.posteriors[:z]))
    ),

	# axis row 3, column 3
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$\\mathbb{E}_{q(\\pi)}[\\pi_k]\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(cop10.history[:π][end])), normalize(probvec(cop10.history[:π][end]) - (10-1)/3*ones(3), 1))
    ),

	# axis row 3, column 4
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$\\mathbb{E}_{q(\\pi)}[\\pi_k]\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(cov10.posteriors[:π])), mean(cov10.posteriors[:π]))
    ),

	
	# axis row 4, column 1
	{
		ybar,
		bar_width=bar_width,
		ylabel_style={align="center"},
		ylabel = "\$\\bm{N=100}\$ \\\\ \\\\ \$p(z=k \\,\\vert\\, y_{1:N})\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(probvec(av100.posteriors[:z])), probvec(av100.posteriors[:z]))
    ),

	# axis row 4, column 2
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$q(z=k)\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(se100.posteriors[:z])), mean(se100.posteriors[:z]))
    ),

	# axis row 4, column 3
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$\\mathbb{E}_{q(\\pi)}[\\pi_k]\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(cop100.history[:π][end])), normalize(probvec(cop100.history[:π][end]) - (100-1)/3*ones(3), 1))
    ),

	# axis row 4, column 4
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$\\mathbb{E}_{q(\\pi)}[\\pi_k]\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(cov100.posteriors[:π])), mean(cov100.posteriors[:π]))
    ),

	
	# axis row 5, column 1
	{
		ybar,
		bar_width=bar_width,
		ylabel_style={align="center"},
		ylabel = "\$\\bm{N=1000}\$ \\\\ \\\\ \$p(z=k \\,\\vert\\, y_{1:N})\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(probvec(av1000.posteriors[:z])), probvec(av1000.posteriors[:z]))
    ),

	# axis row 5, column 2
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$q(z=k)\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(se1000.posteriors[:z])), mean(se1000.posteriors[:z]))
    ),

	# axis row 5, column 3
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$\\mathbb{E}_{q(\\pi)}[\\pi_k]\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(cop1000.history[:π][end])), normalize(probvec(cop1000.history[:π][end]) - (1000-1)/3*ones(3), 1))
    ),

	# axis row 5, column 4
	{
		ybar,
		bar_width=bar_width,
		ylabel = "\$\\mathbb{E}_{q(\\pi)}[\\pi_k]\$",
        style = {thick},
    },
    Plot({ 
			fill="blue",
        },
        Table(1:length(mean(cov1000.posteriors[:π])), mean(cov1000.posteriors[:π]))
    ),
	
)
end

# ╔═╡ 3d9ab3a9-566e-48e3-9f43-3483a25082ec
begin
	pgfsave("../exports/verification_experiments.tikz", fig_tikz)
	pgfsave("../exports/verification_experiments.png", fig_tikz)
	pgfsave("../exports/verification_experiments.pdf", fig_tikz)
end

# ╔═╡ Cell order:
# ╟─5129f69a-43da-4659-9eb4-8d0723eea615
# ╟─e8f2c621-c640-4683-a8bd-d82a9f7dd73f
# ╠═34771c7d-de6a-44c8-aa78-4e991f62f237
# ╠═cd9431ce-ac3c-11ed-02c9-5b5600e798bb
# ╟─ed0dbb47-764e-4dc6-8d58-5623e582692c
# ╟─efd8d329-d6be-4a47-884e-776d7e5f7a43
# ╟─0a8bc1f9-7313-4027-b841-f778b31c91d8
# ╠═34ab39fd-02c3-4e12-857b-3673315cf799
# ╠═b9949a48-89d3-4f31-8d48-375ef6c22f84
# ╟─01060a9a-4cba-4543-924b-2ef9927be903
# ╠═51a5308a-65f7-42d2-90f8-f9fb1de41847
# ╟─cca8814e-600d-4a65-8444-5f7fcfdb5fc8
# ╟─dbbaea7b-4a89-46a4-938c-b1ba1d8a7f7a
# ╠═6666b133-8544-4a71-bc3d-ab59c1bc59ea
# ╠═bf7f3200-1189-4cfe-90c1-3091add478a4
# ╠═9a002685-5228-4245-a834-385c1fe10321
# ╟─b81effdb-e88a-471e-bf64-655e4fb3f2c4
# ╟─0dd9de2b-4047-4cf8-8626-78099d39cb6d
# ╠═2a3928e8-c79c-4471-b2b1-4bc6fcf7dd52
# ╠═584beb0f-b6ae-4323-85e8-15657e82d36a
# ╠═2d83927a-32c0-402e-9f7a-151daef491a5
# ╠═00f83269-9d3b-47b7-87b6-8aa1d8e8cc06
# ╟─bc727e09-6b41-4c6c-906d-5cf94263e164
# ╟─eaefe9da-74ae-461b-bcd6-60d00b8a8c46
# ╟─94316a76-e202-4170-96a1-b6d346aaa806
# ╠═0326ea66-0467-4462-872a-7d701fd14d01
# ╠═1b7af4cb-27a6-4f68-aca5-544d3385efab
# ╠═050018da-382a-45d7-a251-6c4dfb3c1c40
# ╠═222663b1-03a8-4761-aaf1-23b476b037b4
# ╠═af4b9dc8-38d4-48d1-b964-dab9fe6de033
# ╟─af34a2c1-d45e-4245-89fb-e6993fcb968f
# ╟─cf49c906-e2a6-48a4-9fbe-c90aa18bf926
# ╠═4fc49a89-863a-4ad6-b851-df30d76aa1df
# ╠═eb881756-9982-4e78-81e6-2057a5e1c769
# ╟─9434a98e-fef7-4f45-8cfb-5a82e0a1d587
# ╠═cd69f55b-42c1-418e-956c-8da91b629672
# ╠═f1d7aa95-5c64-43ac-8cd9-b131e75f05da
# ╟─876c9a51-4137-4720-94f6-5192403ea217
# ╟─59c7a4e3-54f6-4da9-9257-807578516b09
# ╟─4c891139-3175-4408-9bb8-571aaaa958c0
# ╟─66ad030d-3ccd-438f-b126-c3324707c53e
# ╟─dda9a9e3-6e59-4671-adb3-c7b90144c46d
# ╟─3d9ab3a9-566e-48e3-9f43-3483a25082ec
