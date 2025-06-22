using TikzGraphs
using TikzPictures
using PGFPlotsX
using Plots



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
		labels = [i % 2 == 0 ? labels[i+1] : "" for i ∈ 0:(k-1)]
	end
	if k > 22
		labels = [i % 4 == 0 ? labels[i+1] : "" for i ∈ 0:(k-1)]
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