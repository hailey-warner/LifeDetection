## TODO NEED TO CLEAN:::

using Pkg
Pkg.activate("wandb")
using Wandb
using Logging
using Statistics

using PythonCall


api = Wandb.wandb.Api()

accuracy_avg = []
lambda = []
sensor_usage = []
final_belief = []

#range(0,1,11)
#range(0.99,0.999,10)
for λ in range(0,1,2)#range(0.96,0.99,4)#range(0.55,0.95,5)#range(0.0,1.0,11)
    accuracy_avg_temp = []
    sensor_usage_temp = []
    final_belief_temp = []
    #Sweep_0.05Penalty
    #Sweep_noPenalty
    runs = api.runs("sherpa-rpa/Sweep_0.05Penalty_FalseNegatives_lambda_$(λ)_sample_100_discount0.9")
    for idx in range(1,length(runs)-1)
        if string(runs[idx].state) == "finished"
            print(idx)
            try
                # Append each run's data as a dictionary to the list
                push!(accuracy_avg_temp, runs[idx].summary["accuracy_final"])
                push!(sensor_usage_temp, runs[idx].history(pandas=false)[-1]["Simulation/step"])
                life = "Declare Life" == string(runs[idx].history(pandas=false)[-1]["Simulation/actionName"]) ? 1 : 0
                push!(final_belief_temp, abs(life - pyconvert(Float64, runs[idx].history(pandas=false)[-1]["Simulation/beliefLife"])))
            catch
                println("One run doesn't have error")
            end
        end
    end

    println(accuracy_avg_temp)
    push!(accuracy_avg, mean(accuracy_avg_temp))
    push!(sensor_usage, mean(sensor_usage_temp))
    push!(final_belief, mean(final_belief_temp))
    push!(lambda, λ)

    println(accuracy_avg)
    println(sensor_usage)
    println(lambda)
    println(final_belief)
end

acc_j = [pyconvert(Float64, acc) for acc in accuracy_avg] 
sen_j = [pyconvert(Float64, sen) for sen in sensor_usage] 

p= scatter(sen_j, acc_j, color=:blue, xlabel="Average Sensor Usage over 100 simulations", ylabel="Average Accuracy over 100 simulations", title="Pareto Frontier")
