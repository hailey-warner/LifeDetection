using BayesNets
using TikzGraphs
bn = DiscreteBayesNet()

# life: Boolean (2 states)
push!(bn, DiscreteCPD(:l, [0.9, 0.1]))

# polyelectrolyte: Boolean, parent :l
push!(bn, DiscreteCPD(:pe, [:l], [2], [
    Categorical([0.9, 0.1]),  # l=1
    Categorical([0.1, 0.9]),   # l=2
]))

# chirality: 5 bins, parent :l
push!(
    bn,
    DiscreteCPD(
        :ch,
        [:l],
        [2],
        [
            Categorical(fill(1/5, 5)),                # l=1 (uniform)
            Categorical([0.05, 0.1, 0.2, 0.3, 0.35]),  # l=2 (example)
        ],
    ),
)

# molecular assembly index: Boolean, parent :l
push!(bn, DiscreteCPD(:ma, [:l], [2], [
    Categorical([0.9, 0.1]),  # l=1
    Categorical([0.1, 0.9]),   # l=2
]))

# amino acid abundance: 23 bins, parent :l
w_no = exp.(-0.5 .* (0:22))
p_no = w_no ./ sum(w_no)
w_yes = collect(range(0, stop=0.8, length=23))
p_yes = w_yes ./ sum(w_yes)
push!(bn, DiscreteCPD(:aa, [:l], [2], [
    Categorical(p_no),  # l=1
    Categorical(p_yes),  # l=2
]))

# cell membrane: Boolean, parent :l
push!(bn, DiscreteCPD(:cm, [:l], [2], [
    Categorical([0.9, 0.1]),  # l=1
    Categorical([0.1, 0.9]),   # l=2
]))

# cell autofluorescence: 5 bins, parent :l
push!(
    bn,
    DiscreteCPD(
        :af,
        [:l],
        [2],
        [
            Categorical(fill(1/5, 5)),                # l=1
            Categorical([0.05, 0.1, 0.2, 0.3, 0.35]),  # l=2
        ],
    ),
)

# redox potential: 5 bins, parent :af
push!(bn, DiscreteCPD(:r, [:af], [5], [Categorical(fill(1/5, 5)) for _ = 1:5]))

# salinity: 5 bins, parent :pe
push!(
    bn,
    DiscreteCPD(
        :sal,
        [:pe],
        [2],
        [
            Categorical(fill(1/5, 5)),                # pe=1
            Categorical([0.05, 0.1, 0.2, 0.3, 0.35]),  # pe=2
        ],
    ),
)

# CHNOPS: 5 bins, parent :ma
push!(
    bn,
    DiscreteCPD(
        :chnops,
        [:ma],
        [2],
        [
            Categorical(fill(1/5, 5)),                # ma=1
            Categorical([0.05, 0.1, 0.2, 0.3, 0.35]),  # ma=2
        ],
    ),
)


using StatsBase

# -- Setup --
action_to_cpds = Dict(
    1 => [3, 5, 6, 10],  # HRMS → sal, chnops, r, ma
    2 => [4, 7],
    3 => [4, 7],
    4 => [3, 6],
    5 => [8, 9],
    6 => [2],
)

# Use a specific action
action = 1
var_indices = action_to_cpds[action]
var_names = collect(keys(bn.cpds))
observed_vars = var_names[var_indices]

# Determine domain sizes (bins per variable)
domain_sizes = [length(bn.cpds[v].distributions[1].p) for v in observed_vars]
obs_indices = collect(CartesianIndices(Tuple(domain_sizes)))


posterior_l1 = infer(bn, [:sal, :chnops, :ma, :r], evidence=(Assignment(:l => 1)))
posterior_l2 = infer(bn, [:sal, :chnops, :ma, :r], evidence=(Assignment(:l => 2)))

# Prepare probability arrays
probs_l1 = Float64[]  # P(obs | l=1)
probs_l2 = Float64[]  # P(obs | l=2)

# -- Inference Loop --
for idx in obs_indices
    push!(probs_l1, posterior_l1.potential[idx])
    push!(probs_l2, posterior_l2.potential[idx])
end

# Normalize
probs_l1 ./= sum(probs_l1)
probs_l2 ./= sum(probs_l2)

# Create categorical distributions
dist_l1 = Categorical(probs_l1)  # life = false
dist_l2 = Categorical(probs_l2)  # life = true

# -- Example: Access a specific bin like [1,2,1,2] --
index = LinearIndices(Tuple(domain_sizes))[5, 2, 5, 5]
println("P([1,2,1,2] | l=1): ", probs_l1[index])
println("P([1,2,1,2] | l=2): ", probs_l2[index])




# # Example inference (update evidence values to match new bin indices)
# # using BayesNets: Assignment
# result = infer(bn, :l, evidence=Assignment(:aa => 5, :cm => 1, :ch => 3))
# println(result)

