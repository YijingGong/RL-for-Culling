

def possible_state(state, parity_range, mim_range, mip_range, disease_range):
    if not (state[0] in parity_range and state[1] in mim_range and state[2] in mip_range and state[3] in disease_range):
        return False
    if state[0] == 0: # only springer valid if parity == 0
        if state[1] != 0 or state[2] != 9 or state[3] != 0:
            return False
    else: #parity>=1
        if state[1] == 0 and state[2] == 0:
            return False
        if state[2] == 8 or state[2] == 9: # last two month, must be dry
            if state[1] != 0:
                return False
        else: # if not dry:
            if state[2] != 0: # preg
                if state[1] < state[2]+3: # not possible mim < mip+3 because start breeding during third month
                    return False
    return True

def possible_state2(state, parity_range, mac_range, mip_range, disease_range):
    if not (state[0] in parity_range and state[1] in mac_range and state[2] in mip_range and state[3] in disease_range):
        return False
    if state[0] == 0: # only springer valid if parity == 0
        if state[1] != 0 or state[2] != 9 or state[3] != 0:
            return False
    else: #parity>=1
        if state[1] == 0: # mac cannot be 0 besides (0,0,9)
            return False
        if state[2] != 0: # preg
            if state[1] < state[2]+3: # not possible mim < mip+3 because start breeding during third month
                return False
    return True


def possible_state_no_sick(state, parity_range, mim_range, mip_range):
    if not (state[0] in parity_range and state[1] in mim_range and state[2] in mip_range):
        return False
    if state[0] == 0: # only springer valid if parity == 0
        if state[1] != 0 or state[2] != 9:
            return False
    else: #parity>=1
        if state[1] == 0 and state[2] == 0:
            return False
        if state[2] == 8 or state[2] == 9: # last two month, must be dry
            if state[1] != 0:
                return False
        else: # if not dry:
            if state[2] != 0: # preg
                if state[1] < state[2]+3: # not possible mim < mip+3 because start breeding during third month
                    return False
    return True
