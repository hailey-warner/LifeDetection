using Turing
using Distributions

@model function life_detection_model(evidence)
    # Priors
    l ~ Categorical([0.9, 0.1])  # 1: no life, 2: life

    # polyelectrolyte
    pe_probs = l == 1 ? [0.9, 0.1] : [0.1, 0.9]
    pe ~ Categorical(pe_probs)

    # chirality
    ch_dist = l == 1 ? Beta(1, 1) : Beta(4, 1)
    ch ~ ch_dist

    # molecular assembly index
    ma_probs = l == 1 ? [0.9, 0.1] : [0.1, 0.9]
    ma ~ Categorical(ma_probs)

    # amino acid abundance
    p_no = exp.(-0.5 .* (0:22))
    p_no ./= sum(p_no)
    w_yes = collect(range(0, stop=0.8, length=23))
    p_yes = w_yes ./ sum(w_yes)
    aa_probs = l == 1 ? p_no : p_yes
    aa ~ Categorical(aa_probs)

    # cell membrane
    cm_probs = l == 1 ? [0.9, 0.1] : [0.1, 0.9]
    cm ~ Categorical(cm_probs)

    # cell autofluorescence
    af_dist = l == 1 ? Beta(1, 1) : Beta(4, 1)
    af ~ af_dist

    # Example: observe evidence if provided
    if haskey(evidence, :aa)
        Turing.@addlogprob! logpdf(Categorical(aa_probs), evidence[:aa])
    end
    if haskey(evidence, :cm)
        Turing.@addlogprob! logpdf(Categorical(cm_probs), evidence[:cm])
    end
    if haskey(evidence, :ch)
        Turing.@addlogprob! logpdf(ch_dist, evidence[:ch])
    end
    # Add more evidence as needed...
end

# Example usage:
evidence = Dict(:aa => 6, :cm => 2, :ch => 0.8)  # Note: Categorical indices start at 1
model = life_detection_model(evidence)
chain = sample(model, NUTS(), 1000)

# To get the posterior for l (life):
using StatsPlots
StatsPlots.plot(chain[:l])