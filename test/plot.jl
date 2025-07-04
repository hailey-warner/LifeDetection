#!/usr/bin/env julia

# -------------------------------
# Configuration Flags
# -------------------------------
PLOT_CPDS     = false
PLOT_BN       = false
DECISION_TREE = false

ALPHA_VECTORS_HEATMAP = false
PARETO_FRONTIER_SINGLE = true
PARETO_FRONTIER = false
projname = "hailey_custom"

# -------------------------------
# Package Setup
# -------------------------------
using Pkg
if PLOT_BN || PLOT_CPDS || DECISION_TREE
	Pkg.activate("LifeDetectionPkg")
	Pkg.instantiate()
	include("../src/common/plotting_tikz.jl")
	include("../src/bayes_net.jl")
elseif PARETO_FRONTIER || PARETO_FRONTIER_SINGLE
	Pkg.activate("wandbPkg")
	Pkg.instantiate()
	include("../src/common/plotting_wandb.jl")
	using CSV, DataFrames, Plots
	using Wandb
	using DelimitedFiles
	using Statistics
end


# -------------------------------
# Get PART_ID and NUM_PARTS
# -------------------------------
function get_part_info()
    part_id_str = get(ENV, "PART_ID", get(ARGS, 1, nothing))
    if part_id_str === nothing
        error("Missing PART_ID. Set it via environment variable or pass as CLI argument 1.")
    end
    part_id = try
        parse(Int, part_id_str)
    catch
        error("Invalid PART_ID. Must be an integer.")
    end

    num_parts_str = get(ENV, "NUM_PARTS", get(ARGS, 2, nothing))
    if num_parts_str === nothing
        error("Missing NUM_PARTS. Set it via environment variable or pass as CLI argument 2.")
    end
    num_parts = try
        parse(Int, num_parts_str)
    catch
        error("Invalid NUM_PARTS. Must be an integer.")
    end

    if part_id < 1 || part_id > num_parts
        error("PART_ID must be between 1 and NUM_PARTS (got $part_id, expected 1-$num_parts)")
    end

    return part_id, num_parts
end

# -------------------------------
# PARETO_FRONTIER Collection
# -------------------------------
if PARETO_FRONTIER


	PART_ID, NUM_PARTS = get_part_info()
	# Create unique working directory
	work_dir = "testing/part_$(PART_ID)_$(NUM_PARTS)"
	mkpath(work_dir)

	cd(work_dir) do
			
		entity = "sherpa-rpa"
		# ignore = ["hailey_lambda_0.99_tau_0.05_gamma_0.9_sample_100",
		#           "hailey_lambda_0.925_tau_0.5_gamma_0.9_sample_100"]

		api = Wandb.wandb.Api()
		projects = api.projects(entity=entity)
		all_filtered_projects = [p for p in projects if startswith(string(p.name), projname)]
		total = length(all_filtered_projects)

		chunk_size = ceil(Int, total / NUM_PARTS)
		start_idx = (PART_ID - 1) * chunk_size + 1
		end_idx = min(PART_ID * chunk_size, total)
		filtered_projects = all_filtered_projects[start_idx:end_idx]

		println("PART $PART_ID of $NUM_PARTS handling projects $start_idx to $end_idx")

		count_local = length(filtered_projects)
		average_tt = zeros(Float64, count_local)
		average_ft = zeros(Float64, count_local)
		average_tf = zeros(Float64, count_local)
		average_ff = zeros(Float64, count_local)
		acc_rate = zeros(Float64, count_local)
		errors = fill("", count_local)

		for (i, project) in enumerate(filtered_projects)
			proj_name = string(project.name)
			# try

				runs = api.runs("$entity/$proj_name")
				count = 0
				temp_tt = 0.0
				temp_tf = 0.0
				temp_ft = 0.0
				temp_ff = 0.0
				count_action7 = 0.0

				for run in runs
					if string(run.state) == "finished"
							for hist in run.history(pandas=false)
								# println(hist["action"])
								if 7 == parse(Int,string(hist["action"]))
									count_action7 += 1
								end
							end
							# println("Run $(run.name): action 7 occurred $count_action7 times")
							
						try
							tt = parse(Int, string(run.summary["tt"]))
							tf = parse(Int, string(run.summary["tf"]))
							ft = parse(Int, string(run.summary["ft"]))
							ff = parse(Int, string(run.summary["ff"]))

							total_t = tt + tf
							total_f = ft + ff

							temp_tt += tt / total_t
							temp_tf += tf / total_t
							temp_ft += ft / total_f
							temp_ff += ff / total_f

							count += 1
						catch
							println("Skipping run with missing metrics in $proj_name")
						end
					end
				end

				if count > 0
					average_tt[i] = temp_tt / count
					average_tf[i] = temp_tf / count
					average_ft[i] = temp_ft / count
					average_ff[i] = temp_ff / count
					acc_rate[i] = count_action7 / count
				else
					errors[i] = "No valid runs found"
				end
				println("average_tt: ", average_tt)
				println("average_tf: ", average_tf)
				println("average_ft: ", average_ft)
				println("average_ff: ", average_ff)
				println("acc_rate: ", acc_rate)
			# catch e
			# 	errors[i] = "Error processing $proj_name: $e"
			# end
		end

		# Save results
		if !isdir("pareto_csv")
			mkpath("pareto_csv")
		end

		output_file = "/nfs/gkim65/git_repos/LifeDetection/pareto_csv/pareto_part_$(PART_ID)_of_$(NUM_PARTS).csv"
		open(output_file, "w") do io
			write(io, "project,average_tt,average_tf,average_ft,average_ff,acc_rate,error\n")
			for i in 1:count_local
				proj_str = string(filtered_projects[i].name)
				err = isempty(errors[i]) ? "none" : errors[i]
				write(io, "$proj_str,$(average_tt[i]),$(average_tf[i]),$(average_ft[i]),$(average_ff[i]),$(acc_rate[i]),$err\n")
			end
		end
		println("Saved part results to $output_file")
		
	end
end


if PARETO_FRONTIER_SINGLE

	using Wandb
	api = Wandb.wandb.Api()

	# Replace with your actual entity (user or team name)
	entity = "sherpa-rpa"

	# Get all projects under the entity
	projects = api.projects(entity=entity)

	project_counts = collect(projects)
	# Count the number of projects that start with "pareto_"
	all_filtered_projects = [p for p in projects if startswith(string(p.name), projname)]

	pareto_count = count(project_count -> startswith(string(project_count.name), projname), project_counts)

	average_tt = zeros(Float64,pareto_count)
	average_ft = zeros(Float64,pareto_count)
	average_tf = zeros(Float64,pareto_count)
	average_ff = zeros(Float64,pareto_count)
	acc_rate = zeros(Float64, pareto_count)


	for (proj_idx, project) in enumerate(all_filtered_projects)
		if startswith(string(project.name), projname)
			# Combine the entity and project name into a single string
			project_path = string(entity, "/", project.name)
			runs_pareto = api.runs(project_path)

			count = 0
			temp_tt = 0.0
			temp_tf = 0.0
			temp_ft = 0.0
			temp_ff = 0.0
			count_action7 = 0.0

			for idx in range(1,length(runs_pareto)-1)
				if string(runs_pareto[idx].state) == "finished"
					print(idx)
					for hist in runs_pareto[idx].history(pandas=false)
						# println(hist["action"])
						if 7 == parse(Int,string(hist["action"]))
							count_action7 += 1
						end
					end
					# try
					tt = parse(Int, string(runs_pareto[idx].summary["tt"]))
					tf = parse(Int, string(runs_pareto[idx].summary["tf"]))
					ft = parse(Int, string(runs_pareto[idx].summary["ft"]))
					ff = parse(Int, string(runs_pareto[idx].summary["ff"]))
					total_t = tt + tf
					temp_tt = tt / total_t
					temp_tf = tf / total_t

					total_f = ft + ff
					temp_ft = ft / total_f
					temp_ff = ff / total_f

					count += 1
					# catch
					# 	println("Run doesn't have the correct metrics / data")
					# end
				end
			end 


			average_tt[proj_idx] = temp_tt / count
			average_tf[proj_idx] = temp_tf / count
			average_ft[proj_idx] = temp_ft / count
			average_ff[proj_idx] = temp_ff / count
			acc_rate[proj_idx] = count_action7 / count


		end
	end

	# Scatter plot 1: average_tt vs average_tf
	p1 = scatter(
		average_tf, average_ft,
		xlabel = "false negative: declared dead when life is true",
		ylabel = "false positive: declared life when life is false",
		# title = "Average TT vs Average TF",
		label = "Projects"
	)
	min_tf, min_idx = findmin(average_tf)
	println("Project with smallest average_tf: ", all_filtered_projects[min_idx])
	println(all_filtered_projects)
	println(average_tf)
	println(average_ft)
	# # Scatter plot 2: average_ft vs average_ff
	# p2 = scatter(
	# 	average_ft, average_ff,
	# 	xlabel = "Average FT",
	# 	ylabel = "Average FF",
	# 	title = "Average FT vs Average FF",
	# 	label = "Projects"
	# )

	# Combine plots side by side
	# plot_combined = plot(p1, p2, layout = (1, 2), size = (900, 400))
	savefig(p1, "./figures/pareto_scatter.png")
	display(p1)


	output_file = "pareto_csv/pareto_test.csv"
	open(output_file, "w") do io
		write(io, "project,average_tt,average_tf,average_ft,average_ff,acc_rate\n")
		for i in 1:pareto_count
			proj_str = string(all_filtered_projects[i].name)
			write(io, "$proj_str,$(average_tt[i]),$(average_tf[i]),$(average_ft[i]),$(average_ff[i]),$(acc_rate[i])\n")
		end
	end
	println("Saved part results to $output_file")

end

if PLOT_BN == true
	plot = BayesNets.plot(bn)
	if !isdir("figures")
		mkpath("figures")
	end
	TikzPictures.save(SVG("figures/bayes_net_autogenerated.pdf"), plot)
end

if PLOT_CPDS == true
    @eval begin
		p = @pgf GroupPlot({
			group_style = {group_size = "6 by 4", horizontal_sep = "2.5cm", vertical_sep = "2.5cm"},
			width = "4cm",
			height = "3cm",
		})

		# 1: P(life) prior figure
		prior_plot = make_pgfplot(bn.cpds[bn.name_to_index[:C0]].distributions[1], raw"P($C_0$)")
		PGFPlotsX.pgfsave("figures/prior.png", prior_plot)

		# 2: CPD figure
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C1]].distributions[1], raw"P($C_1$ | $C_0$=false)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C1]].distributions[2], raw"P($C_1$ | $C_0$=true)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C2]].distributions[1], raw"P($C_2$ | $C_0$=false)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C2]].distributions[2], raw"P($C_2$ | $C_0$=true)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C3]].distributions[1], raw"P($C_3$ | $C_0$=false)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C3]].distributions[2], raw"P($C_3$ | $C_0$=true)"))

		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C4]].distributions[1], raw"P($C_4$ | $C_0$=false)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C4]].distributions[2], raw"P($C_4$ | $C_0$=true)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C5]].distributions[1], raw"P($C_5$ | $C_0$=false)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C5]].distributions[2], raw"P($C_5$ | $C_0$=true)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C6]].distributions[1], raw"P($C_6$ | $C_0$=false)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C6]].distributions[2], raw"P($C_6$ | $C_0$=true)"))

		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C7]].distributions[1], raw"P($C_7$ | $C_2$=false)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C7]].distributions[2], raw"P($C_7$ | $C_2$=true)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C8]].distributions[1], raw"P($C_8$ | $C_4$=false, $C_5$=0)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C8]].distributions[23], raw"P($C_8$ | $C_4$=false, $C_5$=22)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C8]].distributions[24], raw"P($C_8$ | $C_4$=true, $C_5$=0)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C8]].distributions[46], raw"P($C_8$ | $C_4$=true, $C_5$=22)"))

		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C9]].distributions[1], raw"P($C_9$ | $C_1$=false, $C_5$=0)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C9]].distributions[23], raw"P($C_9$ | $C_1$=false, $C_5$=22)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C9]].distributions[24], raw"P($C_9$ | $C_1$=true, $C_5$=0)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C9]].distributions[46], raw"P($C_9$ | $C_1$=true, $C_5$=22)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C10]].distributions[1], raw"P($C_{10}$ | $C_5$=0)"))
		push!(p, make_pgfplot(bn.cpds[bn.name_to_index[:C10]].distributions[23], raw"P($C_{10}$ | $C_5$=22)"))

		PGFPlotsX.pgfsave("figures/cpds.png", p)
	end
end
