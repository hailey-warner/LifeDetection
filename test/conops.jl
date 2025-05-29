
function conopsOrbiter(pomdp, o, modeAcc, prevAction)
    sampleVol , lifeState = stateindex_to_state(o, pomdp.lifeStates) 
    if modeAcc
        if sum(pomdp.sampleUse[1:5]) > sampleVol
            return 7, true, 0 # accumulate
        else
            return 1, false, 1 # stop accumulating
        end
    else
        if prevAction < 5
            if pomdp.sampleUse[prevAction+1] < sampleVol
                return prevAction+1, false, prevAction+1
            else
                return 7, true, 0
            end
        else
            # if sum(pomdp.sampleUse[1:5]) > sampleVol
            #     return 1, false, 1
            # else
            return 7, true, 0
            # end
        end
    end
    
end