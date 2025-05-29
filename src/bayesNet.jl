using BayesNets
using TikzGraphs
using TikzPictures
using Plots

PLOT_CPDS = false
PLOT_BN   = false

function JointCategorical(dists::Vector{<:Categorical})
    """
    JointCategorical(dists::Vector{Categorical})

    Takes a vector of `Categorical` distributions over the same support
    and returns a new `Categorical` distribution whose PMF is the normalized
    element-wise product of the input distributions.
    """

    n = length(dists)
    @assert n > 0 "Must provide at least one distribution."

    # get support length
    k = length(dists[1].p)
    @assert all(length(d.p) == k for d in dists) "All distributions must have same support length."

    # initialize with ones
    joint_p = ones(Float64, k)
    
    # multiply component-wise
    for d in dists
        joint_p .*= d.p
    end
    
    # normalize
    joint_p ./= sum(joint_p)

    return Categorical(joint_p)
end

function plot_categorical(cat::Categorical; labels=nothing, title="Categorical Distribution")
    n = length(cat.p)
    labels = labels === nothing ? string.(1:n) : labels
    bar(labels, cat.p, legend=false, title=title)
end

function DiscreteGaussian(μ::Float64, σ::Float64; lo::Float64=0.0, hi::Float64=1.0,  bins::Int=20)
    """
    Discretizes a Gaussian (Normal) distribution with mean `μ` and std `σ` over the range [lo, hi]
    into `n` equal-width bins, and returns a Categorical distribution over those bins.
    """
    # create bin edges and centers
    edges = range(lo, stop=hi, length=bins+1)
    centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

    # evaluate Gaussian PDF at bin centers
    dist = Normal(μ, σ)
    p = pdf.(dist, centers)
    p ./= sum(p)  # normalize

    return Categorical(p)
end

function DiscreteBeta(a, b; bins::Int=10)
    grid = range(0, stop=1, length=bins)
    return Categorical(pdf.(Beta(a, b), grid) ./ sum(pdf.(Beta(a, b), grid)))
end

function DiscreteScaledBeta(a, b; lo::Float64=0.0, hi::Float64=1.0, bins::Int=10)
    grid = range(lo, stop=hi, length=bins)
    scaled = (grid .- lo) ./ (hi - lo)
    return Categorical(pdf.(Beta(a, b), scaled) ./ sum(pdf.(Beta(a, b), scaled)))
end



bn = DiscreteBayesNet()

"""
C0  = life                           (Boolean)
C1  = polyelectrolyte                (Boolean)
C2  = cell membrane                  (Boolean)
C3  = autofluorescence               (Boolean)
C4  = Molecular Assembly Index >= 15 (Boolean)
C5  = Biotic Amino Acid Diversity    (Z ∈ [0, 22])
C6  = L:R Chirality                  (R ∈ [0, 1])
C7  = Salinity                       (R ∈ [0, 1])
C8  = CHNOPS                         (R ∈ [0, 1])
C9  = pH                             (R ∈ [0, 14])
C10 = Redox Potential [V]            (R ∈ [-0.5, 0])

"""

# life: Boolean
push!(bn, DiscreteCPD(:C0, [0.9, 0.1])) # null hypothesis

# polyelectrolyte: Boolean
push!(bn, DiscreteCPD(:C1, [:C0], [2], [
    Categorical([0.95, 0.05]),  # dead
    Categorical([0.15, 0.85])   # alive
]))

# cell membrane: Boolean
push!(bn, DiscreteCPD(:C2, [:C0], [2], [
    Categorical([0.95, 0.05]),  # dead
    Categorical([0.2, 0.8])     # alive
]))

# autofluorescence: Boolean
push!(bn, DiscreteCPD(:C3, [:C0], [2], [
    Categorical([0.95, 0.05]),  # dead
    Categorical([0.1, 0.9])     # alive
]))

# molecular assembly index >= 15: Boolean
push!(bn, DiscreteCPD(:C4, [:C0], [2], [
    Categorical([0.7, 0.3]),  # dead
    Categorical([0.2, 0.8])   # alive
]))

# biotic amino acid diversity: 23 bins
push!(bn, DiscreteCPD(:C5, [:C0], [2], [
    DiscreteScaledBeta(1, 10, lo=0.0, hi=22.0, bins=23), # dead
    DiscreteScaledBeta(3, 1, lo=0.0, hi=22.0, bins=23)   # alive
]))

# chirality: 10 bins
p = [ # U-shaped replacement for Beta(0.5, 0.5)
    4.5, 2.6, 1.7, 1.2, 0.9,
    0.9, 1.2, 1.7, 2.6, 4.5
]
p ./= sum(p)
c6 = Categorical(p)
push!(bn, DiscreteCPD(:C6, [:C0], [2], [
    DiscreteBeta(3, 3, bins=10),  # dead
    c6                            # alive
]))

# salinity: 10 bins
push!(bn, DiscreteCPD(:C7, [:C2], [2], [
    DiscreteBeta(1, 1, bins=5),   # no cell membrane
    DiscreteBeta(4, 1, bins=5)    # cell membrane
]))

# CHNOPS: 10 bins
# C4 x C5 = 2 x 23 = 46 distributions to fully describe CPD
C4_cpds = [
    DiscreteBeta(1, 3, bins=5),  # P(C8 | C4 = 0) (MA < 15)
    DiscreteBeta(5, 1, bins=5)   # P(C8 | C4 = 1) (MA ≥ 15)
]
# P(C8 | C5 = 0 to 22)
C5_cpds = [
    DiscreteScaledBeta(1 + i/22, 1, bins=5) for i in 0:22
]
# Joint distributions: P(C8 | C4, C5) = P(C8 | C4) * P(C8 | C5)
C8_cpds = Categorical[]
for i in 0:1
    for j in 0:22
        joint = JointCategorical([C4_cpds[i+1], C5_cpds[j+1]])
        push!(C8_cpds, joint)
    end
end
push!(bn, DiscreteCPD(:C8, [:C4, :C5], [2, 23], C8_cpds))
#plot_categorical(C8_cpds[30], labels=range(0,1,10), title="CHNOPS Distribution (C4=0, C5=0)")

# pH: 15 bins
C1_cpds = [
    DiscreteGaussian(9.0, 1.5, lo=0.0, hi=14.0, bins=14),
    DiscreteGaussian(10.0, 3.0, lo=0.0, hi=14.0, bins=14)
]
C5_cpds = [
    DiscreteGaussian(μ, σ, lo=0.0, hi=14.0, bins=14)
    for (μ, σ) in zip(range(10.0, stop=7.0, length=23), range(3.0, stop=2.0, length=23))
]
C9_cpds = Categorical[]
for i in 0:1
    for j in 0:22
        joint = JointCategorical([C1_cpds[i+1], C5_cpds[j+1]])
        push!(C9_cpds, joint)
    end
end
push!(bn, DiscreteCPD(:C9, [:C1, :C5], [2, 23], C9_cpds))

# redox potential: 10 bins
C5_cpds = [
    DiscreteScaledBeta(3.0, 10.0 - 7.0 * i / 22, lo=-0.5, hi=0.0, bins=10)
    for i in 0:22
]
push!(bn, DiscreteCPD(:C10, [:C5], [23], C5_cpds))

if PLOT_BN == true
    plot = BayesNets.plot(bn)
    TikzPictures.save(SVG("./figures/bayes_net_autogenerated"), plot)
end
if PLOT_CPDS == true
    using Plots
    plots = Plots.Plot[]  # store subplots here

    function make_plot(dist, title)
        k = length(dist.p)
        labels = k == 2 ? ["false", "true"] : string.(0:k-1)
        if k > 11
            labels = [i % 2 == 0 ? labels[i+1] : "" for i in 0:k-1]
        end
        return bar(dist.p, legend=false, title=title, xticks=(1:k, labels))
    end

    # --- C0
    push!(plots, make_plot(bn.cpds[bn.name_to_index[:C0]].distributions[1], "P(C0)"))

    # --- C1
    push!(plots, make_plot(bn.cpds[bn.name_to_index[:C1]].distributions[1], "P(C1 | C0=false)"))
    push!(plots, make_plot(bn.cpds[bn.name_to_index[:C1]].distributions[2], "P(C1 | C0=true)"))

    # --- C2 to C6
    for i in 2:6
        push!(plots, make_plot(bn.cpds[bn.name_to_index[Symbol(:C, i)]].distributions[1], "P(C$i | C0=false)"))
        push!(plots, make_plot(bn.cpds[bn.name_to_index[Symbol(:C, i)]].distributions[2], "P(C$i | C0=true)"))
    end

    # --- C7
    push!(plots, make_plot(bn.cpds[bn.name_to_index[:C7]].distributions[1], "P(C7 | C2=false)"))
    push!(plots, make_plot(bn.cpds[bn.name_to_index[:C7]].distributions[2], "P(C7 | C2=true)"))

    # --- C8
    for (c4, c5) in ((false,0), (false,22), (true,0), (true,22))
        idx = 1 + c4*23 + c5
        push!(plots, make_plot(bn.cpds[bn.name_to_index[:C8]].distributions[idx], "P(C8 | C4=$c4, C5=$c5)"))
    end

    # --- C9
    for (c1, c5) in ((false,0), (false,22), (true,0), (true,22))
        idx = 1 + c1*23 + c5
        push!(plots, make_plot(bn.cpds[bn.name_to_index[:C9]].distributions[idx], "P(C9 | C1=$c1, C5=$c5)"))
    end

    # --- C10
    push!(plots, make_plot(bn.cpds[bn.name_to_index[:C10]].distributions[1], "P(C10 | C5=0)"))
    push!(plots, make_plot(bn.cpds[bn.name_to_index[:C10]].distributions[23], "P(C10 | C5=22)"))


    # custom layout: 7 columns x 6 rows = 42 slots
    # `_` for blanks
    custom_layout = @layout [
        a1  a2  a3  a4  a5  a6  a7
        _   b2  b3  b4  b5  b6  b7
        _   _   _   _   _   _   _
        _   c2  c3  c4  c5  _   _
        _   d2  d3  d4  d5  _   _
        _   _   e3  e4  _   _   _
        _   _   f3  f4  _   _   _
    ]

    # order plots into correct layout positions
    ordered = vcat(
        plots[1],
        plots[2:2:13],   
        plots[3:2:14],
        plots[14],
        plots[16],
        plots[20],
        plots[24],
        plots[15],
        plots[17],
        plots[21],
        plots[25],
        plots[18],
        plots[22],
        plots[19],
        plots[23]
    )
    fig = Plots.plot(ordered..., layout=custom_layout, size=(2000, 1600))
    savefig(fig, "./figures/cpds.png")
end

# debugging:
# for cpd in bn.cpds
#     println("Checking ", cpd.target)
#     f = convert(Factor, cpd)
# end

function distObservations(actionCpds, lifeState, action, maxObs)

    posterior = infer(bn, actionCpds[action] ,evidence=(Assignment(:C0 => lifeState )))

    probs = Float64[]  # P(obs | l=1)

    # Determine domain sizes (bins per variable)
    domain_sizes = [length(bn.cpds[bn.name_to_index[v]].distributions[1].p) for v in posterior.dimensions]
    obs_indices = collect(CartesianIndices(Tuple(domain_sizes)))

    # -- Inference Loop --
    for idx in obs_indices
        push!(probs, posterior.potential[idx])
    end

    while length(probs) < maxObs
        push!(probs, 0)
    end

    probs ./= sum(probs)

    return probs#, Categorical(probs)
end


function determineMaxObs(actionCpds)

    maxObs = 0
    
    for action in collect(keys(actionCpds))
        for lifeState in [1,2]
            probs = Float64[]  # P(obs | l=1)
            posterior = infer(bn, actionCpds[action] ,evidence=(Assignment(:C0 => lifeState )))

            # Determine domain sizes (bins per variable)
            domain_sizes = [length(bn.cpds[bn.name_to_index[v]].distributions[1].p) for v in posterior.dimensions]
            obs_indices = collect(CartesianIndices(Tuple(domain_sizes)))

            # -- Inference Loop --
            for idx in obs_indices
                push!(probs, posterior.potential[idx])
            end
            if length(probs) > maxObs
                maxObs = length(probs)
            end
        end
    end

    return maxObs
end


