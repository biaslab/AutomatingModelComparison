### A Pluto.jl notebook ###
# v0.19.26

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

# ╔═╡ 0a8bc1f9-7313-4027-b841-f778b31c91d8
md"""number of samples: $(@bind nr_samples Slider(1:2_000; default=1500, show_value=true))"""

# ╔═╡ 34ab39fd-02c3-4e12-857b-3673315cf799
dist = MixtureModel(Normal, [(-3.0, 1.0), (0.0, 1.0), (4.0, 1.0)], [0.2, 0.5, 0.3]);

# ╔═╡ b9949a48-89d3-4f31-8d48-375ef6c22f84
noise_var = 5.0

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

    # instantiate variables
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
        θ1[i] ~ NormalMeanPrecision(-3, 1)
        θ2[i] ~ NormalMeanPrecision(0, 1)
        θ3[i] ~ NormalMeanPrecision(4, 1)

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

    # instantiate variables
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
        θ1[i] ~ NormalMeanPrecision(-3, 1)
        θ2[i] ~ NormalMeanPrecision(0, 1)
        θ3[i] ~ NormalMeanPrecision(4, 1)

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

    msgs_names = Val{(name(switchinterface), name(inputsinterfaces[1]))}()
    msgs_observable =
    combineLatest((ReactiveMP.messagein(switchinterface), combineLatest(map((input) -> ReactiveMP.messagein(input), inputsinterfaces), PushNew())), PushNew()) |>
        map_to((ReactiveMP.messagein(switchinterface), ReactiveMP.ManyOf(map((input) -> ReactiveMP.messagein(input), inputsinterfaces))))
    return msgs_names, msgs_observable
end

# create an observable that is used to compute the output with pipeline constraints
function ReactiveMP.get_messages_observable(factornode::MixtureNode{N, F, Nothing, ReactiveMP.FactorNodePipeline{P, EmptyPipelineStage}}, messages::Tuple{NTuple{N, ReactiveMP.IndexedNodeInterface}}) where {N, F <: FullFactorisation, P <: EnforceMarginalFunctionalDependency}
    inputsinterfaces = messages[1]

    msgs_names = Val{(name(inputsinterfaces[1]), )}()
    msgs_observable =
    combineLatest(map((input) -> ReactiveMP.messagein(input), inputsinterfaces), PushNew()) |>
        map_to((ReactiveMP.ManyOf(map((input) -> ReactiveMP.messagein(input), inputsinterfaces)),))
    return msgs_names, msgs_observable
end

# create an observable that is used to compute the input with pipeline constraints
function ReactiveMP.get_messages_observable(factornode::MixtureNode{N, F, Nothing, ReactiveMP.FactorNodePipeline{P, EmptyPipelineStage}}, messages::Tuple{ReactiveMP.NodeInterface}) where {N, F <: FullFactorisation, P <: EnforceMarginalFunctionalDependency}
    outputinterface = messages[1]

    msgs_names = Val{(name(outputinterface), )}()
    msgs_observable = combineLatestUpdates((ReactiveMP.messagein(outputinterface), ), PushNew())
    return msgs_names, msgs_observable
end

function ReactiveMP.get_marginals_observable(factornode::MixtureNode{N, F, Nothing, ReactiveMP.FactorNodePipeline{P, EmptyPipelineStage}}, marginals::Tuple{ReactiveMP.NodeInterface}) where {N, F <: FullFactorisation, P <: EnforceMarginalFunctionalDependency}
    switchinterface = marginals[1]

    marginal_names       = Val{(name(switchinterface), )}()
    marginals_observable = combineLatestUpdates((getmarginal(ReactiveMP.connectedvar(switchinterface), IncludeAll()), ), PushNew())

    return marginal_names, marginals_observable
end
	
end

# ╔═╡ 0326ea66-0467-4462-872a-7d701fd14d01
@model function model_combination()

    # instantiate variables
    y = datavar(Float64)
    α = datavar(Vector{Float64})
	
    # specify initial distribution over clusters
    π ~ Dirichlet(α)
		
	# prior over model selection variable
	z ~ Categorical(π) where { pipeline = EnforceMarginalFunctionalDependency(:out) }

	# specify prior models over θ
	θ1 ~ NormalMeanPrecision(-3, 1)
	θ2 ~ NormalMeanPrecision(0, 1)
	θ3 ~ NormalMeanPrecision(4, 1)

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
		initmarginals = (π = Dirichlet(ones(3)./3*length(data)*100), ),
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
	plt.bar(1:length(probvec(results_combination.history[:π][end])), normalize(probvec(results_combination.history[:π][end]) - ones(3)./3*nr_samples*100, 1))
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

    # instantiate variables
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
		θ1[i] ~ NormalMeanPrecision(-3, 1)
		θ2[i] ~ NormalMeanPrecision(0, 1)
		θ3[i] ~ NormalMeanPrecision(4, 1)
	
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
		axx.bar(1:length(probvec(co.history[:π][end])), normalize(probvec(co.history[:π][end]) - ones(3)./3*(100*k-1), 1))
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
		width = "1.75in",
		height = "1.75in",
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
        Table(1:length(mean(cop1.history[:π][end])), normalize(probvec(cop1.history[:π][end]) - (1*100-1)/3*ones(3), 1))
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
        Table(1:length(mean(cop5.history[:π][end])), normalize(probvec(cop5.history[:π][end]) - (5*100-1)/3*ones(3), 1))
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
        Table(1:length(mean(cop10.history[:π][end])), normalize(probvec(cop10.history[:π][end]) - (10*100-1)/3*ones(3), 1))
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
        Table(1:length(mean(cop100.history[:π][end])), normalize(probvec(cop100.history[:π][end]) - (100*100-1)/3*ones(3), 1))
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
        Table(1:length(mean(cop1000.history[:π][end])), normalize(probvec(cop1000.history[:π][end]) - (1000*100-1)/3*ones(3), 1))
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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PGFPlotsX = "8314cec4-20b6-5062-9cdb-752b83310925"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RxInfer = "86711068-29c9-4ff7-b620-ae75d7495b3d"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[compat]
Distributions = "~0.25.95"
PGFPlotsX = "~1.6.0"
PlutoUI = "~0.7.51"
PyPlot = "~2.11.1"
RxInfer = "~2.11.0"
SpecialFunctions = "~2.2.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "70311c4ca3abd960d74a067dbd892414a37a39e7"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "d3f758863a47ceef2248d136657cb9c033603641"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.8"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4aff5fa660eb95c2e0deb6bcdabe4d9a96bc4667"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "0.8.18"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "89e0654ed8c7aebad6d5ad235d6242c2d737a928"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.3"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "02d2316b7ffceff992f3096ae48c7829a8aa0638"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.3"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "e32a90da027ca45d84678b826fffd3110bb3fc90"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.8.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefaultApplication]]
deps = ["InteractiveUtils"]
git-tree-sha1 = "c0dfa5a35710a193d83f03124356eef3386688fc"
uuid = "3f0dd361-4fe0-5fc6-8523-80b14ec94d85"
version = "1.1.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "c72970914c8a21b36bbc244e9df0ed1834a0360b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.95"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DomainIntegrals]]
deps = ["CompositeTypes", "DomainSets", "FastGaussQuadrature", "GaussQuadrature", "HCubature", "IntervalSets", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "0b0425701a4b9b0b9da831d834e2580af35b321a"
uuid = "cc6bae93-f070-4015-88fd-838f9505a86c"
version = "0.4.3"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "698124109da77b6914f64edd696be8dccf90229e"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.6.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "0f478d8bad6f52573fb7658a263af61f3d96e43a"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.5.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7072f1e3e5a8be51d525d64f63d3ec1287ff2790"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.11"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "c6e4a1fbe73b31a3dea94b1da449503b8830c306"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.21.1"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GaussQuadrature]]
deps = ["SpecialFunctions"]
git-tree-sha1 = "eb6f1f48aa994f3018cbd029a17863c6535a266d"
uuid = "d54b0c1a-921d-58e0-8e36-89d8069c0969"
version = "0.5.8"

[[deps.GraphPPL]]
deps = ["MacroTools", "TupleTools"]
git-tree-sha1 = "36d1953626dcb87e87824488167a5a27e6046424"
uuid = "b3f8163a-e979-4e85-b43e-1f63d8c8b42c"
version = "3.1.0"

[[deps.HCubature]]
deps = ["Combinatorics", "DataStructures", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "e95b36755023def6ebc3d269e6483efa8b2f7f65"
uuid = "19dc6840-f33b-545b-b366-655c7e3ffd49"
version = "1.5.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "734fd90dd2f920a2f1921d5388dcebe805b262dc"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.14"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "84204eae2dd237500835990bcade263e27674a93"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.16"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "88b8f66b604da079a627b6fb2860d3704a6729a1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.14"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "MatrixFactorizations", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "7402f6be1a28a05516c6881596879e6d18d28039"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "0.22.18"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "3bb62b5003bc7d2d49f26663484267dc49fa1bf5"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.159"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "0ff59b4b9024ab9a736db1ad902d2b1b48441c19"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "0.9.6"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "a89b11f0f354f06099e4001c151dffad7ebab015"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.5"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.PGFPlotsX]]
deps = ["ArgCheck", "Dates", "DefaultApplication", "DocStringExtensions", "MacroTools", "OrderedCollections", "Parameters", "Requires", "Tables"]
git-tree-sha1 = "3e7a0345b9f37da2cd770a5d47bb5cb6e62c7a81"
uuid = "8314cec4-20b6-5062-9cdb-752b83310925"
version = "1.6.0"

    [deps.PGFPlotsX.extensions]
    ColorsExt = "Colors"
    ContourExt = "Contour"
    MeasurementsExt = "Measurements"
    StatsBaseExt = "StatsBase"

    [deps.PGFPlotsX.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Contour = "d38c429a-6771-53c6-b99e-75d170b6e991"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a5aef8d4a6e8d81f171b2bd4be5265b01384c74c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.10"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "259e206946c293698122f63e2b513a7c99a244e8"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "62f417f6ad727987c755549e9cd88c46578da562"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.95.1"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "92e7ca803b579b8b817f004e74b205a706d9a974"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.11.1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.ReactiveMP]]
deps = ["DataStructures", "Distributions", "DomainIntegrals", "DomainSets", "FastGaussQuadrature", "ForwardDiff", "HCubature", "LazyArrays", "LinearAlgebra", "LoopVectorization", "MacroTools", "Optim", "PositiveFactorizations", "Random", "Rocket", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "TinyHugeNumbers", "TupleTools", "Unrolled"]
git-tree-sha1 = "437cbb6ea40b99e52d8144b32c5c4d68a7528d7a"
uuid = "a194aa59-28ba-4574-a09c-4a745416d6e3"
version = "3.9.0"

    [deps.ReactiveMP.extensions]
    ReactiveMPOptimisersExt = "Optimisers"
    ReactiveMPRequiresExt = "Requires"
    ReactiveMPZygoteExt = "Zygote"

    [deps.ReactiveMP.weakdeps]
    Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.Rocket]]
deps = ["DataStructures", "Sockets", "Unrolled"]
git-tree-sha1 = "33e270ce5710d5315f28c205ec7d598c4fdf660d"
uuid = "df971d30-c9d6-4b37-b8ff-e965b2cb3a40"
version = "1.7.0"

[[deps.RxInfer]]
deps = ["DataStructures", "Distributions", "DomainSets", "GraphPPL", "LinearAlgebra", "MacroTools", "Optim", "ProgressMeter", "Random", "ReactiveMP", "Reexport", "Rocket", "TupleTools"]
git-tree-sha1 = "2de7d688c27e5d9e27369570696dd9193f9396f6"
uuid = "86711068-29c9-4ff7-b620-ae75d7495b3d"
version = "2.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "4b8586aece42bee682399c4c4aee95446aa5cd19"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.39"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "dbde6766fc677423598138a5951269432b0fcc90"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.7"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "33040351d2403b84afce74dae2e22d3f5b18edcb"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "8982b3607a212b070a5e46eea83eb62b4744ae12"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.25"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "c97f60dd4f2331e1a495527f80d242501d2f9865"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.1"

[[deps.TinyHugeNumbers]]
git-tree-sha1 = "d1bd5b57d45431fcbf2db38d3e17453a603e76ad"
uuid = "783c9a47-75a3-44ac-a16b-f1ab7b3acf04"
version = "1.0.0"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Unrolled]]
deps = ["MacroTools"]
git-tree-sha1 = "6cc9d682755680e0f0be87c56392b7651efc2c7b"
uuid = "9602ed7d-8fef-5bc8-8597-8f21381861e8"
version = "0.1.5"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "b182207d4af54ac64cbc71797765068fdeff475d"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.64"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─5129f69a-43da-4659-9eb4-8d0723eea615
# ╟─e8f2c621-c640-4683-a8bd-d82a9f7dd73f
# ╠═cd9431ce-ac3c-11ed-02c9-5b5600e798bb
# ╟─ed0dbb47-764e-4dc6-8d58-5623e582692c
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
