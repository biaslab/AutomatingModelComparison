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

# ╔═╡ 0dd5f0f0-ad35-11ed-1f87-89c9e84b49c5
# remove once merged with master
using Pkg; Pkg.activate("..");

# ╔═╡ 71406015-1c45-4d78-bbcb-30506d6c68ac
using RxInfer, Random, PlutoUI, PyPlot, LaTeXStrings, PGFPlotsX

# ╔═╡ f99478a1-20ed-405a-8405-77b9d087ca4e
begin
	using SpecialFunctions: besselk
	
	anchor = nothing
	function productlogpdf(mx, vx, my, vy, rho; truncation=10, jitter=1e-20)
	    
	    # construct logpdf function
	    logpdf = function (x)
	        
	        # add jitter
	        x += jitter
	        
	        # first term
	        term1 = -1/(2*(1-rho^2)) * (mx^2/vx + my^2/vy - 2*rho*(x + mx*my)/sqrt(vx*vy))
	        
	        # other terms
	        term2 = 0.0
	        for n = 0:truncation
	            for m = 0:2*n
	                term2 += x^(2*n - m) * abs(x)^(m - n) * sqrt(vx)^(m - n - 1) / 
	                    (pi * factorial(2*n) * (1 - rho^2)^(2*n + 1/2) * sqrt(vy)^(m - n + 1)) *
	                    (mx / vx - rho * my / sqrt(vx * vy) )^m *
	                    binomial(2*n, m) * 
	                    (my / vy - rho*mx/sqrt(vx*vy))^(2 * n - m) * 
	                    besselk( m-n, abs(x) / (( 1 - rho^2) * sqrt(vx*vy)) )
	            end
	        end
	        
	        # return logpdf
	        return term1 + log(term2)
	
	    end
	
	    # return logpdf
	    return logpdf
	    
	end
	
	@rule typeof(*)(:out, Marginalisation) (m_A::UnivariateGaussianDistributionsFamily, m_in::UnivariateGaussianDistributionsFamily, meta::TinyCorrection) = begin 
	    
	    @logscale 0
	
	    # extract statistics
	    μ_in, var_in = mean_var(m_in)
	    μ_A, var_A = mean_var(m_A)
	
	    # logpdf function
	    logpdf = productlogpdf(μ_in, var_in, μ_A, var_A, 0)
	
	    # return logpdf message
	    return ContinuousUnivariateLogPdf(logpdf)
	end

	@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::ContinuousUnivariateLogPdf) = begin 
	
	    @logscale 0
	    # construct shifted logpdf
	    shifted_logpdf = (x) -> logpdf(m_in2, mean(m_out) - x)
	    # return shifted logpdf
	    return ContinuousUnivariateLogPdf(shifted_logpdf)
	end

	function RxInfer.ReactiveMP.prod(::ProdAnalytical, left::ContinuousUnivariateLogPdf, right::PointMass)
	    return right
	end

	function RxInfer.ReactiveMP.multiply_addons(left_addons::Tuple{AddonLogScale{Nothing}}, ::Nothing, ::PointMass, left_message::ContinuousUnivariateLogPdf, right_message::PointMass)
	    return AddonLogScale(left_message.logpdf(mean(right_message)))
	end
	
end

# ╔═╡ db9fda8c-38e3-457f-9a6a-9554e4c9ddeb
md"""
# Validation experiments: mixed model
"""

# ╔═╡ be9de7f5-9ce1-4209-afa2-96de8e99dc2b
begin
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}");
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{bm}");
end;

# ╔═╡ a2c160ae-6536-45cc-b56c-cf804799eeb8
md"""
### Generate data
"""

# ╔═╡ 513f9123-a620-4918-a47e-742c6bdd225c
@bind nr_samples Slider(1:2_000; default=1500, show_value=true)

# ╔═╡ 1cbf3d0a-1f7b-4f92-a644-318e3ea037d6
data = randn(MersenneTwister(123), nr_samples);

# ╔═╡ f8640cf6-b911-404c-8231-86d76e0dcd77
begin
	f1(x) = productlogpdf(0, 1.0, 0.5, 1.0, 0)(x+0.2)
	f2(x) = productlogpdf(0, 1.0, 0.5, 1.0, 0)(x-0.9)
	f3(x) = productlogpdf(0, 1.0, 0.5, 1.0, 0)(x+1.8)
	plt.figure()
	plt.hist(data, bins=100, density=true)
	plt.plot(collect(-5:0.01:5), exp.(f1.(collect(-5:0.01:5))), color="red")
	plt.plot(collect(-5:0.01:5), exp.(f2.(collect(-5:0.01:5))), color="red")
	plt.plot(collect(-5:0.01:5), exp.(f3.(collect(-5:0.01:5))), color="red")
	plt.xlabel(L"x")
	plt.ylabel("number of samples")
	plt.ylim(0, 1)
	plt.xlim(-3, 3)
	plt.grid()
	plt.gcf()
end

# ╔═╡ 36e84806-8879-4a0a-a6d1-0afc92404766
md"""
### Model averaging
"""

# ╔═╡ 7e6983c8-1af1-4755-9123-2b1377bdaf7e
@model function mixed_model_averaging(nr_samples)

	# instantiate variables
	y = datavar(Float64, nr_samples)
	a = randomvar(nr_samples)
	b = randomvar(nr_samples)
	c = randomvar(nr_samples)
	d = randomvar(nr_samples)

	z ~ Categorical(ones(3) ./ 3)
	
	for n in 1:nr_samples
	
	    d[n] ~ Mixture(z, (-0.2, 0.9, -1.8))
	
	    a[n] ~ Normal(mean = 0.5, variance = 1.0)
	    b[n] ~ Normal(mean = 0.0, variance = 1.0)
	    c[n] ~ a[n] * b[n]
	
	    y[n] ~ c[n] + d[n]

	end

	return y
	
end

# ╔═╡ 5e4bff9f-8848-4b47-9a1d-da421de25563
begin
	anchor
	function run_averaging(data)
		return inference(
	    	model = mixed_model_averaging(length(data)), 
	 		data  = (y = data, ), 
			addons = AddonLogScale(),
	    	returnvars = (z = KeepLast(), )
		)
	end
end

# ╔═╡ f39148b9-99ec-4a43-98b3-9a491fb8350f
results_averaging = run_averaging(data)

# ╔═╡ 79928b5f-8a45-40fe-be2f-fc396c0df441
begin
	plt.figure()
	plt.bar([-0.2, 0.9, -1.8], probvec(results_averaging.posteriors[:z]))
	plt.grid()
	plt.ylabel(L"p(z\mid y)")
	plt.xlabel(L"d")
	plt.xticks([-0.2, 0.9, -1.8])
	plt.xlim(-2.5, 2.5)
	plt.gcf()
end

# ╔═╡ e911254e-3200-4398-bddd-1c55589f3db9
md"""
### Model combination (variational)
"""

# ╔═╡ dad183c6-22cb-43d8-a1e9-85c71cc3f6ef
@model function mixed_model_combination(nr_samples)

	# instantiate variables
	y = datavar(Float64, nr_samples)
	a = randomvar(nr_samples)
	b = randomvar(nr_samples)
	c = randomvar(nr_samples)
	d = randomvar(nr_samples)
	z = randomvar(nr_samples)

    π ~ Dirichlet(ones(3) ./ 3)
		
	for n in 1:nr_samples

		z[n] ~ Categorical(π)
	
	    d[n] ~ Mixture(z[n], (-0.2, 0.9, -1.8))
	
	    a[n] ~ Normal(mean = 0.5, variance = 1.0)
	    b[n] ~ Normal(mean = 0.0, variance = 1.0)
	    c[n] ~ a[n] * b[n]
	
	    y[n] ~ c[n] + d[n]

	end

	return y
	
end

# ╔═╡ 28446695-9b11-4071-92e7-983df2f9c294
@constraints function constraints_combination()
    q(π, z) = q(π)q(z)
end

# ╔═╡ 583fe420-629f-433d-9690-64da95d14968
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

# ╔═╡ 6d498ed3-0eef-4aea-9b6f-8ccf4770b33b
 function run_combination(data)
	 tmp
	 return inference(
		 model         = mixed_model_combination(length(data)),
		 data          = (y = data, ),
		 constraints   = constraints_combination(),
		 initmarginals = (π = vague(Dirichlet, 3), z = vague(Categorical, 3)),
		 returnvars    = (π=KeepLast(), z=KeepLast()),
		 addons        = AddonLogScale(),
		 iterations    = 10
	)
 end

# ╔═╡ d2ff1764-a587-4857-b4f4-62cdaca3fd3d
results_combination = run_combination(data)

# ╔═╡ e2660fe1-2314-4b6b-89ab-04d5bba617be
begin
	plt.figure()
	plt.bar([-0.2, 0.9, -1.8], mean(results_combination.posteriors[:π]), yerr=sqrt.(var(results_combination.posteriors[:π])))
	plt.xlabel(L"k")
	plt.ylabel(L"p(z=k)")
	plt.xticks([-0.2, 0.9, -1.8])
	plt.xlim(-2.5, 2.5)
	plt.grid()
	plt.gcf()
end

# ╔═╡ f2c3f4b4-86c3-40f3-bce5-fbecec9f658b
begin
	plt.figure(figsize=(15,5))
	plt.scatter(data, zeros(nr_samples).+rand(nr_samples), c=argmax.(probvec.(results_combination.posteriors[:z])))
	plt.xlim(-3,3)
	plt.grid()
	plt.gcf()
end

# ╔═╡ 09864b89-5aaf-49ba-b610-45062596fc4a
begin
	plt.figure()
	plt.hist(data, bins=100, density=true, alpha=0.5)
	plt.plot(collect(-5:0.01:5), mean(results_combination.posteriors[:π])[1]*exp.(f1.(collect(-5:0.01:5))) + mean(results_combination.posteriors[:π])[2]*exp.(f2.(collect(-5:0.01:5))) + mean(results_combination.posteriors[:π])[3]*exp.(f3.(collect(-5:0.01:5))), color="blue", linewidth=2)
	plt.plot(collect(-5:0.01:5), mean(results_combination.posteriors[:π])[1]*exp.(f1.(collect(-5:0.01:5))), color="red", linestyle="--")
	plt.plot(collect(-5:0.01:5), mean(results_combination.posteriors[:π])[2]*exp.(f2.(collect(-5:0.01:5))), color="red", linestyle="--")
	plt.plot(collect(-5:0.01:5), mean(results_combination.posteriors[:π])[3]*exp.(f3.(collect(-5:0.01:5))), color="red", linestyle="--")
	plt.xlabel(L"x")
	plt.ylabel("number of samples")
	plt.ylim(0, 1)
	plt.xlim(-3, 3)
	plt.grid()
	plt.gcf()
end

# ╔═╡ 94c8a3ef-ff30-4bdd-ac4c-bfa49fe9bb8a
md"""
### Overview
"""

# ╔═╡ d6497f63-3c45-49f9-bf02-e5e016628b24
begin
	data_1000 = randn(MersenneTwister(123), 1000);
	results_averaging_1000 = run_averaging(data_1000)
	results_combination_1000 = run_combination(data_1000)
	
fig_tikz = @pgf GroupPlot(

	# group plot options
	{
		group_style = {
			group_size = "2 by 2",
			horizontal_sep = "1cm",
		},
		label_style={font="\\footnotesize"},
		ticklabel_style={font="\\scriptsize",},
        grid = "major",
		width = "3in",
		height = "3in",
		# ylabel_shift = "-5pt",
		# xlabel_shift = "-5pt",
		# xlabel = "\$k\$",
	},

	# row 1, column 1
	{
		ybar,
		bar_width="20pt",
		ylabel = "\$p(z\\mid y)\$",
		xlabel = "\$d\$",
        style = {thick},
		title = "\\textbf{Model averaging}\\\\",
		title_style={align="center"},
		ymin = 0
    },
    Plot({ 
			fill="blue",
        },
        Table([-0.2, 0.9, -1.8], probvec(results_averaging_1000.posteriors[:z]))
    ),

	# row 1, column 2
	{
		ybar,
		bar_width="20pt",
		ylabel = "\$\\mathbb{E}_{q(\\pi)}[\\pi_k]\$",
        style = {thick},
		title = "\\textbf{Model combination}\\\\(variational)",
		title_style={align="center"},
		ymin = 0
    },
    Plot({ 
			fill="blue",
        },
        Table([-0.2, 0.9, -1.8], mean(results_combination_1000.posteriors[:π]))
    ),

	# row 2, column 1
	{
		# ybar,
		# interval,
		# bar_width="20pt",
		# ylabel = "\$\\mathbb{E}_{q(\\pi)}[\\pi_k]\$",
        # style = {thick},
		# title = "\\textbf{Model combination}\\\\(variational)",
		# title_style={align="center"},
		ymin = 0
    },
    Plot({ 
			fill="blue",
			hist={bins=100,},
        },
        Table(data_1000)
    ),
	
	
)
end

# ╔═╡ Cell order:
# ╟─db9fda8c-38e3-457f-9a6a-9554e4c9ddeb
# ╠═0dd5f0f0-ad35-11ed-1f87-89c9e84b49c5
# ╠═71406015-1c45-4d78-bbcb-30506d6c68ac
# ╟─be9de7f5-9ce1-4209-afa2-96de8e99dc2b
# ╟─f99478a1-20ed-405a-8405-77b9d087ca4e
# ╟─a2c160ae-6536-45cc-b56c-cf804799eeb8
# ╟─513f9123-a620-4918-a47e-742c6bdd225c
# ╠═1cbf3d0a-1f7b-4f92-a644-318e3ea037d6
# ╟─f8640cf6-b911-404c-8231-86d76e0dcd77
# ╟─36e84806-8879-4a0a-a6d1-0afc92404766
# ╠═7e6983c8-1af1-4755-9123-2b1377bdaf7e
# ╠═5e4bff9f-8848-4b47-9a1d-da421de25563
# ╠═f39148b9-99ec-4a43-98b3-9a491fb8350f
# ╟─79928b5f-8a45-40fe-be2f-fc396c0df441
# ╟─e911254e-3200-4398-bddd-1c55589f3db9
# ╠═dad183c6-22cb-43d8-a1e9-85c71cc3f6ef
# ╠═28446695-9b11-4071-92e7-983df2f9c294
# ╟─583fe420-629f-433d-9690-64da95d14968
# ╠═6d498ed3-0eef-4aea-9b6f-8ccf4770b33b
# ╠═d2ff1764-a587-4857-b4f4-62cdaca3fd3d
# ╟─e2660fe1-2314-4b6b-89ab-04d5bba617be
# ╟─f2c3f4b4-86c3-40f3-bce5-fbecec9f658b
# ╟─09864b89-5aaf-49ba-b610-45062596fc4a
# ╟─94c8a3ef-ff30-4bdd-ac4c-bfa49fe9bb8a
# ╠═d6497f63-3c45-49f9-bf02-e5e016628b24
