using Pkg
Pkg.activate("wandbPkg")
Pkg.instantiate()
using CSV, DataFrames, Plots

const NUM_PARTS = 200  # ← change if you used a different number

function combine_pareto_results(NUM_PARTS)

    df_all = DataFrame()

    for i in 1:NUM_PARTS
        fname = "pareto_csv/pareto_part_$(i)_of_$(NUM_PARTS).csv"
        if isfile(fname)
            df = CSV.read(fname, DataFrame)
            df_all = vcat(df_all, df)
        else
            println("Warning: missing $fname")
        end
    end

    CSV.write("pareto_csv/pareto_combined.csv", df_all)

    for col in [:average_tf, :average_ft]
        if col in names(df_all)
            replace!(df_all[!, col], x -> isnan(x) ? 0.0 : x)
        end
    end

    p = scatter(
        df_all.average_tf,
        df_all.average_ft,
        xlabel = "False Negative (declared dead when life = true)",
        ylabel = "False Positive (declared life when life = false)",
        title = "Pareto Frontier Summary",
        legend = false,
        xlims = (-0.1, 1.1),
        ylims = (-0.1, 1.1),
        markersize = 5
    )

    savefig(p, "figures/pareto_scatter_combined.png")
    display(p)
end

combine_pareto_results(NUM_PARTS)
