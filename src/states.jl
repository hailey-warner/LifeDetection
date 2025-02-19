
function Base.length(mdp::LifeDetectionMDP)
    return 0 # of states
end

POMDPs.states(mdp::LifeDetectionMDP) = []

POMDPs.initialstate(mdp::LifeDetectionMDP) = []

# probability of life [0, 100]
# SMS health [boolean]
# HRMS health [boolean]
# ESA health [boolean]
# µCE-LIF health [boolean]
# nanopore health [boolean]
# microscope health [boolean]


# pH & redox potential <-- Nernst equation


    """
    Na⁺ (sodium): Major salt component, detected in plumes
    Cl⁻ (chloride): Primary anion, forms salts
    HCO₃⁻ (bicarbonate): pH buffer, carbon source
    CO₃²⁻ (carbonate): pH buffer, carbon source
    H⁺ (hydrogen ion): Critical for pH and redox
    HS⁻/H₂S (hydrogen sulfide): Energy source for potential life
    """

