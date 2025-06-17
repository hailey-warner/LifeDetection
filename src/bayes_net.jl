using BayesNets

include("common/bayesnet_generation.jl")


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
C5_cpds = [DiscreteBeta(1 + i/22, 1, bins=5) for i ∈ 0:22]
# Joint distributions: P(C8 | C4, C5) = P(C8 | C4) * P(C8 | C5)
C8_cpds = Categorical[]
for i ∈ 0:1
	for j ∈ 0:22
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
for i ∈ 0:1
	for j ∈ 0:22
		joint = JointCategorical([C1_cpds[i+1], C5_cpds[j+1]])
		push!(C9_cpds, joint)
	end
end
push!(bn, DiscreteCPD(:C9, [:C1, :C5], [2, 23], C9_cpds))

# redox potential: 10 bins
C5_cpds = [DiscreteBeta(3.0, 10.0 - 7.0 * i / 22, lo=-0.5, hi=0.0, bins=10) for i ∈ 0:22]
push!(bn, DiscreteCPD(:C10, [:C5], [23], C5_cpds))


