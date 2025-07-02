using Pkg
Pkg.activate("wandbPkg")
Pkg.instantiate()
using CSV, DataFrames, Plots, Statistics

const NUM_PARTS = 100  # ← change if needed


function combine_pareto_results(NUM_PARTS)
    df_all = DataFrame()

    combined_csv = "pareto_csv/pareto_combined.csv"
    if isfile(combined_csv)
        println("Combined CSV already exists, skipping CSV read and write.")
        df_all = CSV.read(combined_csv, DataFrame)
        # return df_all
    else
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
    end

    for col in [:average_tf, :average_ft]
        if col in names(df_all)
            replace!(df_all[!, col], x -> isnan(x) ? 0.0 : x)
        end
    end

    # --- Pareto Frontier Scatter ---
    # Ensure "figures" directory exists
    if !isdir("figures")
        mkpath("figures")
    end

    println(df_all.average_tf)
    println(df_all.average_ft)
    # Get indices where average_ft < 0.5
    idxs = findall(<(0.5), df_all.average_ft)
    # Get corresponding values from average_ft and average_tf
    ft_below_05 = df_all.average_ft[idxs]
    tf_matching = df_all.average_tf[idxs]
    println(ft_below_05)
    println(tf_matching)
    println("average_ft < 0.5: ", ft_below_05)
    println("Corresponding average_tf: ", tf_matching)
    p1 = scatter(
        tf_matching,#df_all.average_tf,
        ft_below_05,#df_all.average_ft,
        xlabel = "False Negative (declared dead when life = true)",
        ylabel = "False Positive (declared life when life = false)",
        title = "Pareto Frontier Summary",
        legend = false,
        xlims = (-0.1, 1.1),
        ylims = (-0.1, 0.1),
        markersize = 5,
        
    )
    display(p1)
    savefig(p1, "figures/pareto_scatter_combined.png")
    

    precision = [tt / (tt + ft) for (tt, ft) in zip(df_all.average_tt, df_all.average_ft)]
    recall = [tt / (tt + tf) for (tt, tf) in zip(df_all.average_tt, df_all.average_tf)]
    # --- Precision-Recall Curve ---
    # Update axis labels to reflect precision and recall
    p2 = scatter(
        df_all.average_ft,
        df_all.average_tf,
        xlabel = "Precision",
        ylabel = "Recall",
        title = "Precision-Recall Curve",
        legend = false,
        xlims = (-0.1, 1.1),
        ylims = (-0.1, 1.1),
        markersize = 5
    )
    # Update axis labels to reflect precision and recall

    savefig(p2, "figures/PR_curve.png")
    display(p2)
    average_tf = filter(x -> !isnan(x) && x != 0 && x != 1, df_all.average_tf)
    println(sort(average_tf))
    println("Smallest average_tf value: ", minimum(average_tf))
    p3 = histogram(
        average_tf,
        bins = 100,
        xlabel = "average_tf",
        ylabel = "Frequency",
        title = "Histogram of average_tf",
        legend = false
    )
    savefig(p3, "figures/average_tf_histogram.png")
    display(p3)
    println(average_tf)
    p4 = scatter(
        average_tf,
        average_tf,
        xlabel = "average_tf",
        ylabel = "average_tf",
        title = "average_tf vs average_tf",
        legend = false,
        xlims = (-0.1, 1.1),
        ylims = (-0.1, 1.1),
        markersize = 5
    )
    savefig(p4, "figures/average_tf_vs_average_tf.png")
    display(p4)

    p5 = scatter(
        average_tf,
        df_all.average_ft[1:length(average_tf)],
        xlabel = "average_tf",
        ylabel = "average_ft",
        title = "average_tf vs average_ft",
        legend = false,
        xlims = (-0.1, 1.1),
        ylims = (-0.1, 1.1),
        markersize = 5
    )
    savefig(p5, "figures/averagetf.png")
    display(p5)
end


combine_pareto_results(NUM_PARTS)
