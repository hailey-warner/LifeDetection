using BayesNets
using Distributions
using Plots

bn = BayesNet()

# life: Boolean
push!(bn, StaticCPD(:l, Categorical([0.9,0.1]))) # null hypothesis H0

# polyelectrolyte: Boolean                           l=0             l=1
push!(bn, CategoricalCPD{Bernoulli}(:pe, [:l], [2], [Bernoulli(0.1), Bernoulli(0.9)]))

# chirality: R -> [0, 1]
push!(bn, CategoricalCPD{Beta}(:ch, [:l], [2], [Beta(1,1), Beta(4,1)]))

# molecular assembly index: Boolean
push!(bn, CategoricalCPD{Bernoulli}(:ma, [:l], [2], [Bernoulli(0.1), Bernoulli(0.9)]))

# amino acid abundance: Z -> [0, 22]
w_no = exp.(-0.5.*0:22)
p_no = w_no ./ sum(w_no)
w_yes = collect(range(0, stop=0.8, length=23))
p_yes = w_yes ./ sum(w_yes)
push!(bn, CategoricalCPD{Categorical}(:aa, [:l], [2], [Categorical(p_no), Categorical(p_yes)]))

# cell membrane: Boolean
push!(bn, CategoricalCPD{Bernoulli}(:cm, [:l], [2], [Bernoulli(0.1), Bernoulli(0.9)]))

# cell autofluorescence: R -> [0, 1]
push!(bn, CategoricalCPD{Beta}(:af, [:l], [2], [Beta(1,1), Beta(4,1)]))

# pH: R -> [1, 14] ***NOT CORRECT DIST***
#push!(bn, CategoricalCPD{Bernoulli}(:ph, [:ch, :pe], [2,2], [Bernoulli(0.1), Bernoulli(0.2), Bernoulli(1.0), Bernoulli(0.4)]))

# redox potential: R -> [-0.5, 0.0] [V]
push!(bn, CategoricalCPD{UnivariateDistribution}(:r, [:af], [2], [Distributions.Uniform(-0.5, 0.0), LocationScale(-0.5, 0.5, Beta(2, 1))]))

# salinity: R -> [0, 1]
push!(bn, CategoricalCPD{Beta}(:sal, [:pe], [2], [Beta(1,1), Beta(1,2)]))

# CHNOPS: R -> [0, 1]
push!(bn, CategoricalCPD{Beta}(:chnops, [:ma], [2], [Beta(1,1), Beta(4,1)]))

# ...existing code...
result = infer(GibbsSamplingNodewise(),bn, :l, Dict(:aa => 5, :cm => 1, :ch => 1))
println(result)
# ...existing code...

typeof(bn)