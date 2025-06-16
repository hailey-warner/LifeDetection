using BayesNets
using TikzGraphs
using TikzPictures
using Plots
using PGFPlotsX

PLOT_CPDS = true
PLOT_BN = false

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

function DiscreteGaussian(
    μ::Float64,
    σ::Float64;
    lo::Float64=0.0,
    hi::Float64=1.0,
    bins::Int=20,
)
    """
    Discretizes a Gaussian (Normal) distribution with mean `μ` and std `σ` over the range [lo, hi]
    into equal-width bins, and returns a Categorical distribution over those bins.
    """
    # create bin edges and centers
    edges = range(lo, stop=hi, length=bins+1)
    centers = (edges[1:(end-1)] .+ edges[2:end]) ./ 2

    # evaluate Gaussian PDF at bin centers
    dist = Normal(μ, σ)
    p = pdf.(dist, centers)
    p ./= sum(p)  # normalize

    return Categorical(p)
end

function DiscreteBeta(α, β; lo::Float64=0.0, hi::Float64=1.0, bins::Int=10)
    """
    Discretizes a Beta distribution over the range [lo, hi]
    into equal-width bins, and returns a Categorical distribution over those bins.
    """
    grid = range(lo, stop=hi, length=bins)
    scaled = (grid .- lo) ./ (hi - lo) # scale domain to [a, b]
    p = pdf.(Beta(α, β), scaled)
    p ./= sum(p) # normalize

    return Categorical(p)
end

function plot_categorical(
    cat::Categorical;
    labels=nothing,
    title="Categorical Distribution",
)
    """
    Plots a Categorical distribution (for debugging).
    """
    n = length(cat.p)
    labels = labels === nothing ? string.(1:n) : labels
    bar(labels, cat.p, legend=false, title=title)
end

function make_pgfplot(dist, title)
    """
    Helper function to create individual pgf plots.
    """
    k = length(dist.p)
    labels = k == 2 ? ["false", "true"] : string.(0:(k-1))
    if k > 11 && k < 22
        labels = [i % 2 == 0 ? labels[i+1] : "" for i = 0:(k-1)]
    end
    if k > 22
        labels = [i % 4 == 0 ? labels[i+1] : "" for i = 0:(k-1)]
    end

    # Calculate bar width to fill space exactly
    axis_width_cm = 5.0
    bar_width_cm = axis_width_cm / (2*k)
    x_min = 0.5
    x_max = k + 0.5

    return @pgf Axis(
        {
            "ybar",
            "title" = title,
            "xtick" = 1:k,
            "xticklabels" = labels,
            "axis x line" = "bottom",
            "axis y line" = "left",
            "xticklabel style" = "{rotate=45, anchor=east}",
            "ymin" = 0,
            "width" = "$(axis_width_cm)cm",
            "height" = "4cm",
            "bar width" = "$(bar_width_cm)cm",
            "xmin" = x_min,
            "xmax" = x_max,
            "enlarge x limits" = "0.1",
            "y tick label style" = "{/pgf/number format/fixed, /pgf/number format/precision=2}",
            "ylabel style" = "{align=center}",
            "scaled y ticks" = "false",
            "axis y line*" = "left",
            "axis x line*" = "bottom",
            "every axis y label/.style" = "{at={(ticklabel cs:0)},rotate=90,anchor=center}",
            "every axis x label/.style" = "{at={(ticklabel cs:0.5)},anchor=north}",
            "xlabel near ticks",
            "ylabel near ticks",
            "tick align" = "outside",
            "tick pos" = "left",
        },
        Plot(
            {"mark" = "none", "color" = "blue", "thick"},
            Coordinates([(i, p) for (i, p) in enumerate(dist.p)]),
        ),
    )
end

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
bn = DiscreteBayesNet()

# life: Boolean
push!(bn, DiscreteCPD(:C0, [0.9, 0.1])) # null hypothesis

# polyelectrolyte: Boolean
push!(
    bn,
    DiscreteCPD(:C1, [:C0], [2], [
        Categorical([0.95, 0.05]),  # dead
        Categorical([0.15, 0.85]),   # alive
    ]),
)

# cell membrane: Boolean
push!(
    bn,
    DiscreteCPD(:C2, [:C0], [2], [
        Categorical([0.95, 0.05]),  # dead
        Categorical([0.2, 0.8]),     # alive
    ]),
)

# autofluorescence: Boolean
push!(
    bn,
    DiscreteCPD(:C3, [:C0], [2], [
        Categorical([0.95, 0.05]),  # dead
        Categorical([0.1, 0.9]),     # alive
    ]),
)

# molecular assembly index >= 15: Boolean
push!(bn, DiscreteCPD(:C4, [:C0], [2], [
    Categorical([0.7, 0.3]),  # dead
    Categorical([0.2, 0.8]),   # alive
]))

# biotic amino acid diversity: 23 bins
push!(
    bn,
    DiscreteCPD(
        :C5,
        [:C0],
        [2],
        [
            DiscreteBeta(1, 10, lo=0.0, hi=22.0, bins=23), # dead
            DiscreteBeta(3, 1, lo=0.0, hi=22.0, bins=23),   # alive
        ],
    ),
)

# chirality: 10 bins
p = [ # U-shaped replacement for Beta(0.5, 0.5)
    4.5,
    2.6,
    1.7,
    1.2,
    0.9,
    0.9,
    1.2,
    1.7,
    2.6,
    4.5,
]
p ./= sum(p)
c6 = Categorical(p)
push!(bn, DiscreteCPD(:C6, [:C0], [2], [
    DiscreteBeta(3, 3, bins=10),  # dead
    c6,                            # alive
]))

# salinity: 10 bins
push!(
    bn,
    DiscreteCPD(:C7, [:C2], [2], [
        DiscreteBeta(1, 1, bins=5),   # no cell membrane
        DiscreteBeta(4, 1, bins=5),    # cell membrane
    ]),
)

# CHNOPS: 10 bins
# C4 x C5 = 2 x 23 = 46 distributions to fully describe CPD
C4_cpds = [
    DiscreteBeta(1, 3, bins=5),  # P(C8 | C4 = 0) (MA < 15)
    DiscreteBeta(5, 1, bins=5),   # P(C8 | C4 = 1) (MA ≥ 15)
]
# P(C8 | C5 = 0 to 22)
C5_cpds = [DiscreteBeta(1 + i/22, 1, bins=5) for i = 0:22]
# Joint distributions: P(C8 | C4, C5) = P(C8 | C4) * P(C8 | C5)
C8_cpds = Categorical[]
for i = 0:1
    for j = 0:22
        joint = JointCategorical([C4_cpds[i+1], C5_cpds[j+1]])
        push!(C8_cpds, joint)
    end
end
push!(bn, DiscreteCPD(:C8, [:C4, :C5], [2, 23], C8_cpds))
#plot_categorical(C8_cpds[30], labels=range(0,1,10), title="CHNOPS Distribution (C4=0, C5=0)")

# pH: 15 bins
C1_cpds = [
    DiscreteGaussian(9.0, 1.5, lo=0.0, hi=14.0, bins=14),
    DiscreteGaussian(10.0, 3.0, lo=0.0, hi=14.0, bins=14),
]
C5_cpds = [
    DiscreteGaussian(μ, σ, lo=0.0, hi=14.0, bins=14) for
    (μ, σ) in zip(range(10.0, stop=7.0, length=23), range(3.0, stop=2.0, length=23))
]
C9_cpds = Categorical[]
for i = 0:1
    for j = 0:22
        joint = JointCategorical([C1_cpds[i+1], C5_cpds[j+1]])
        push!(C9_cpds, joint)
    end
end
push!(bn, DiscreteCPD(:C9, [:C1, :C5], [2, 23], C9_cpds))

# redox potential: 10 bins
C5_cpds = [DiscreteBeta(3.0, 10.0 - 7.0 * i / 22, lo=-0.5, hi=0.0, bins=10) for i = 0:22]
push!(bn, DiscreteCPD(:C10, [:C5], [23], C5_cpds))



if PLOT_BN == true
    plot = BayesNets.plot(bn)
    PGFPlotsX.save("figures/bayes_net_autogenerated.pdf", plot)
end

if PLOT_CPDS == true
    p = @pgf GroupPlot({
        group_style =
            {group_size = "6 by 4", horizontal_sep = "2.5cm", vertical_sep = "2.5cm"},
        width = "4cm",
        height = "3cm",
    })

    # 1: P(life) prior figure
    prior_plot =
        make_pgfplot(bn.cpds[bn.name_to_index[:C0]].distributions[1], raw"P($C_0$)")
    PGFPlotsX.pgfsave("figures/prior.png", prior_plot)

    # 2: CPD figure
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C1]].distributions[1],
            raw"P($C_1$ | $C_0$=false)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C1]].distributions[2],
            raw"P($C_1$ | $C_0$=true)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C2]].distributions[1],
            raw"P($C_2$ | $C_0$=false)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C2]].distributions[2],
            raw"P($C_2$ | $C_0$=true)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C3]].distributions[1],
            raw"P($C_3$ | $C_0$=false)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C3]].distributions[2],
            raw"P($C_3$ | $C_0$=true)",
        ),
    )

    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C4]].distributions[1],
            raw"P($C_4$ | $C_0$=false)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C4]].distributions[2],
            raw"P($C_4$ | $C_0$=true)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C5]].distributions[1],
            raw"P($C_5$ | $C_0$=false)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C5]].distributions[2],
            raw"P($C_5$ | $C_0$=true)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C6]].distributions[1],
            raw"P($C_6$ | $C_0$=false)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C6]].distributions[2],
            raw"P($C_6$ | $C_0$=true)",
        ),
    )

    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C7]].distributions[1],
            raw"P($C_7$ | $C_2$=false)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C7]].distributions[2],
            raw"P($C_7$ | $C_2$=true)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C8]].distributions[1],
            raw"P($C_8$ | $C_4$=false, $C_5$=0)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C8]].distributions[23],
            raw"P($C_8$ | $C_4$=false, $C_5$=22)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C8]].distributions[24],
            raw"P($C_8$ | $C_4$=true, $C_5$=0)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C8]].distributions[46],
            raw"P($C_8$ | $C_4$=true, $C_5$=22)",
        ),
    )

    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C9]].distributions[1],
            raw"P($C_9$ | $C_1$=false, $C_5$=0)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C9]].distributions[23],
            raw"P($C_9$ | $C_1$=false, $C_5$=22)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C9]].distributions[24],
            raw"P($C_9$ | $C_1$=true, $C_5$=0)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C9]].distributions[46],
            raw"P($C_9$ | $C_1$=true, $C_5$=22)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C10]].distributions[1],
            raw"P($C_{10}$ | $C_5$=0)",
        ),
    )
    push!(
        p,
        make_pgfplot(
            bn.cpds[bn.name_to_index[:C10]].distributions[23],
            raw"P($C_{10}$ | $C_5$=22)",
        ),
    )

    PGFPlotsX.pgfsave("figures/cpds.png", p)
end
