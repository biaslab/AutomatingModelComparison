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

# ╔═╡ d1688d30-37ea-49b9-bc71-876d6a4bfe5b
# remove once merged with master
using Pkg; Pkg.activate("..");

# ╔═╡ ee518def-a08b-47f2-8363-8af95bbbacf8
using RxInfer, Distributions, PlutoUI, PyPlot, PGFPlotsX, LaTeXStrings, Random, LinearAlgebra, HDF5, DSP

# ╔═╡ fbcf63ae-b37d-11ed-2179-9912c0e834fe
md"""
# Validation experiments: voice activity detection
"""

# ╔═╡ b6e4a912-aae6-40cb-8a3f-f66bf6d5ef07
md"""
### Data loading
"""

# ╔═╡ 982c9d79-4214-45f8-9b8f-803b5614a534
begin
	# fetch data
	signal = h5read("../data/0000000001.h5", "signal")
	
	# plot signal
	plt.figure(figsize=(10, 4))
	plt.plot(collect(1:length(signal))./ 16000, signal)
	plt.grid()
	plt.xlabel("time [sec]")
	plt.gcf()
end

# ╔═╡ 198c343c-fc80-46c0-b3d4-93a9f86368a9
md"""
### Model specification
"""

# ╔═╡ f4f024dd-f94e-40e3-b09d-bec4f83658b4
begin
md"""ρ: $(@bind ρ Slider(0:0.01:1; default=(mean(signal[9001:28001] ./ signal[9000:28000])), show_value=true))

σ2: $(@bind σ2 Slider(0:0.01:1; default=(var(signal[9001:28001] - ρ .* signal[9000:28000])), show_value=true))
"""
end

# ╔═╡ aaae6b26-3c77-409e-adfb-07d8dc8b8354
@model function model_VAD(ρ, σ2, zp, T)

    # specify observed signal
    y = datavar(Float64)
	
	μ_s = datavar(Float64)
	Σ_s = datavar(Float64)
	π   = datavar(Vector{Float64})

    # speech model
    s_prev ~ NormalMeanVariance(μ_s, Σ_s)
    s_new ~ NormalMeanVariance(ρ * s_prev, σ2)

    # noise model
    n_new ~ NormalMeanVariance(0.0, 0.01)

    # selection variable
    z_prev ~ Categorical(π)
	z_new ~ Transition(z_prev, T)

    # specify mixture prior Distribution
    x ~ Mixture(z_new, (s_new, n_new))

    # specify observation noise
    y ~ NormalMeanPrecision(x, 1000.0)
    
    return y, z_new, s_new
end

# ╔═╡ cdb18485-8bd9-4cbe-91a7-3668a6ac7d5b
md"""
### Probabilistic inference
"""

# ╔═╡ 3c365005-d998-4554-a6af-5ac0c1974502
autoupdates_VAD = @autoupdates begin
	μ_s, Σ_s = mean_var(q(s_new))
	π = probvec(q(z_new))
end;

# ╔═╡ 10839358-e190-42e4-b9d8-005710d5a4b6
 function run_VAD(data, ρ, σ2, pz, T)
	 return rxinference(
		model         = model_VAD(ρ, σ2, pz, T),
		data          = (y = data, ),
		autoupdates   = autoupdates_VAD,
		initmarginals = (s_new = vague(NormalMeanVariance), z_new = vague(Categorical, 2)),
		returnvars    = (:s_new, :z_new),
		keephistory   = length(signal),
		historyvars   = (s_new = KeepLast(), z_new = KeepLast()),
		autostart     = true,
		addons        = AddonLogScale()
	)
 end;

# ╔═╡ b6fc7ba2-5906-4704-bf4e-ddf7ab6e9e83
md"""
### Results
"""

# ╔═╡ 5bb50a56-8cb2-4f15-9007-b2e7f37348e4
begin
T_switch = 1e-5
T = [1-T_switch T_switch; T_switch 1-T_switch]
end

# ╔═╡ a6acfd4c-78b0-4b36-bf9b-9ee3eda0e5a6
results_VAD = run_VAD(signal, ρ, σ2, [0.5, 0.5], T)

# ╔═╡ f76a4388-87bf-4c68-8c2c-00f0d7b53102
begin
	# plot signal
	fig, ax1 = plt.subplots(figsize=(10,4))
	plt.plot(collect(1:length(signal))./ 16000, signal)
	plt.grid()
	plt.xlabel("time [sec]")
	plt.ylabel("signal")
	ax2 = ax1.twinx()
	plt.plot(collect(1:length(signal)-100)./ 16000, -(conv(argmax.(probvec.(results_VAD.history[:z_new])), 1/100*ones(100))[100:end-100]).+2, c="orange")
	ax2.set_ylim([-0.05, 1.05])
	ax2.set_yticks([0, 1], ["noise", "speech"])
	plt.gcf()
end

# ╔═╡ 55efc3bb-1c98-4227-bef9-d590a5974ca5


# ╔═╡ Cell order:
# ╟─fbcf63ae-b37d-11ed-2179-9912c0e834fe
# ╠═d1688d30-37ea-49b9-bc71-876d6a4bfe5b
# ╠═ee518def-a08b-47f2-8363-8af95bbbacf8
# ╟─b6e4a912-aae6-40cb-8a3f-f66bf6d5ef07
# ╟─982c9d79-4214-45f8-9b8f-803b5614a534
# ╟─198c343c-fc80-46c0-b3d4-93a9f86368a9
# ╟─f4f024dd-f94e-40e3-b09d-bec4f83658b4
# ╠═aaae6b26-3c77-409e-adfb-07d8dc8b8354
# ╟─cdb18485-8bd9-4cbe-91a7-3668a6ac7d5b
# ╠═3c365005-d998-4554-a6af-5ac0c1974502
# ╠═10839358-e190-42e4-b9d8-005710d5a4b6
# ╟─b6fc7ba2-5906-4704-bf4e-ddf7ab6e9e83
# ╠═5bb50a56-8cb2-4f15-9007-b2e7f37348e4
# ╠═a6acfd4c-78b0-4b36-bf9b-9ee3eda0e5a6
# ╟─f76a4388-87bf-4c68-8c2c-00f0d7b53102
# ╠═55efc3bb-1c98-4227-bef9-d590a5974ca5
