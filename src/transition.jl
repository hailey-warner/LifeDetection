using Match

PROBABILITY_DAMAGE = Dict(:SMS => 0.01,
                          :HRMS => 0.01,
                          :ESA => 0.01,
                          :µCE_LIF => 0.01,
                          :nanopore => 0.10,
                          :microscope => 0.05)

function transition(mdp::LifeDetectionMDP, s, a)
    next_states = []
    
    @match a begin
        :melt_sample => # Handle melting sample
        :vaporize_sample => # Handle vaporizing sample
        :flush_sample => # Handle flushing sample
        :run_SMS => # Handle SMS analysis
        :run_HRMS => # Handle HRMS analysis
        :run_ESA => # Handle ESA analysis
        :run_µCE_LIF => # Handle µCE-LIF analysis
        :run_nanopore => # Handle nanopore analysis
        :run_microscope => # Handle microscope analysis
        _ => error("Unknown action: $a")
    end
    
    return next_states
end


