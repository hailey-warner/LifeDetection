using BayesNets
using TikzGraphs
bn = DiscreteBayesNet()

# life: Boolean (2 states)
push!(bn, DiscreteCPD(:l, [0.9, 0.1]))

# polyelectrolyte: Boolean, parent :l
push!(bn, DiscreteCPD(:pe, [:l], [2], [
    Categorical([0.9, 0.1]),  # l=1
    Categorical([0.1, 0.9])   # l=2
]))

# chirality: 5 bins, parent :l
push!(bn, DiscreteCPD(:ch, [:l], [2], [
    Categorical(fill(1/5, 5)),                # l=1 (uniform)
    Categorical([0.05, 0.1, 0.2, 0.3, 0.35])  # l=2 (example)
]))

# molecular assembly index: Boolean, parent :l
push!(bn, DiscreteCPD(:ma, [:l], [2], [
    Categorical([0.9, 0.1]),  # l=1
    Categorical([0.1, 0.9])   # l=2
]))

# amino acid abundance: 23 bins, parent :l
w_no = exp.(-0.5 .* (0:22))
p_no = w_no ./ sum(w_no)
w_yes = collect(range(0, stop=0.8, length=23))
p_yes = w_yes ./ sum(w_yes)
push!(bn, DiscreteCPD(:aa, [:l], [2], [
    Categorical(p_no),  # l=1
    Categorical(p_yes)  # l=2
]))

# cell membrane: Boolean, parent :l
push!(bn, DiscreteCPD(:cm, [:l], [2], [
    Categorical([0.9, 0.1]),  # l=1
    Categorical([0.1, 0.9])   # l=2
]))

# cell autofluorescence: 5 bins, parent :l
push!(bn, DiscreteCPD(:af, [:l], [2], [
    Categorical(fill(1/5, 5)),                # l=1
    Categorical([0.05, 0.1, 0.2, 0.3, 0.35])  # l=2
]))

# redox potential: 5 bins, parent :af
push!(bn, DiscreteCPD(:r, [:af], [5], [
    Categorical(fill(1/5, 5)) for _ in 1:5
]))

# salinity: 5 bins, parent :pe
push!(bn, DiscreteCPD(:sal, [:pe], [2], [
    Categorical(fill(1/5, 5)),                # pe=1
    Categorical([0.05, 0.1, 0.2, 0.3, 0.35])  # pe=2
]))

# CHNOPS: 5 bins, parent :ma
push!(bn, DiscreteCPD(:chnops, [:ma], [2], [
    Categorical(fill(1/5, 5)),                # ma=1
    Categorical([0.05, 0.1, 0.2, 0.3, 0.35])  # ma=2
]))

# # Example inference (update evidence values to match new bin indices)
# # using BayesNets: Assignment
# result = infer(bn, :l, evidence=Assignment(:aa => 5, :cm => 1, :ch => 3))
# println(result)