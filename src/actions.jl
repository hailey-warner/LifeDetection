POMDPs.actions(mdp::LifeDetectionMDP) = [:melt_sample,
                                         :vaporize_sample,
                                         :flush_sample,
                                         :run_SMS,
                                         :run_HRMS
                                         :run_ESA,
                                         :run_µCE_LIF,
                                         :run_nanopore,
                                         :run_microscope,
                                         ]

POMDPs.actionindex(mdp::LifeDetectionMDP, a::Symbol) = findfirst(==(a), ACTIONS)
