
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
