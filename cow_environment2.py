# change state of MIM to MAC (month after calving)
import numpy as np
from scipy.integrate import quad
import random
import utility
import animal_constants

class CowEnv:
    def __init__(self, parity_range, mac_range, mip_range, disease_range):
        # only consider parity, MAC, MIP, disease
        self.state = (0, 0, 9, 0)
        self.parameter_a = self.parameter_b = self.parameter_c = None
        self.actions = ['Keep', 'Cull']
        self.parity_range = parity_range
        self.mac_range = mac_range
        self.mip_range = mip_range
        self.disease_range = disease_range

    def reset(self):
        self.state = (random.choice(self.parity_range), random.choice(self.mac_range), random.choice(self.mip_range), random.choice(self.disease_range)) 
        while utility.possible_state2(self.state, self.parity_range, self.mac_range, self.mip_range, self.disease_range) == False:
            self.state = (random.choice(self.parity_range), random.choice(self.mac_range), random.choice(self.mip_range), random.choice(self.disease_range)) 
        return self.state

    def step(self, action):
        slaughter_income = 0
        calf_income = 0
        milk_income = 0
        breed_cost = 0
        treatment_cost = 0
        parity, mac, mip, disease = self.state
        # print("state:", parity, mac, mip, disease)

        if action == 'Cull':
            slaughter_income = self.slaughter(parity, disease) 
            self.state = (0, 0, 9, 0) # replaced by a new springer
            reward = slaughter_income - animal_constants.REPLACEMENT_COST 
            # print(">cull")
            # print("slaughter_income:", slaughter_income, "replacement_cost", animal_constants.REPLACEMENT_COST )
        
        else: 
            next_parity = parity # by default, parity does not change, unless mip == 9
            next_mac = 0
            next_mip = 0
            next_mip = 0
            next_disease = 0

            # death
            if self.death(parity, disease): # died
                # print(">keep died")
                slaughter_income = 0 # it's 0 because it is a died cow
                self.state = (0, 0, 9, 0) # replaced by a new springer
                reward = slaughter_income - animal_constants.REPLACEMENT_COST 
                # print("slaughter_income:", slaughter_income, "replacement_cost", animal_constants.REPLACEMENT_COST )
            else:
                # milking
                if mac != 0 and mip != 7 and mip != 8:
                    dim = (mac-1)*30 + 1 
                    self.parameter_a, self.parameter_b, self.parameter_c = self.assign_woods_parameters(parity)
                    milk_production = self.calc_integral_wood_curve(dim, dim+30, self.parameter_a, self.parameter_b, self.parameter_c)
                    milk_income = milk_production*2.2/100 * animal_constants.MILK_PRICE  
                next_mac = mac + 1

                # pregnancy status
                if mip == 9: # calving
                    calf_income = animal_constants.CALF_PRICE
                    next_parity = parity+1
                    next_mac = 1
                    next_mip = 0
                elif mip == 0: # breeding
                    if mac>=3:
                        breed_cost = animal_constants.BREED_COST_PER_MONTH  
                        if self.breed(parity, mac, disease) == True:
                            next_mip = 1
                        else: 
                            next_mip = 0
                else: # keep pregnancy
                    next_mip = mip + 1

                # Disease affect slauter income (in slaughter() function), milk income, breed success (in breed() function), and treatment_cost 
                if disease == 1: # when the cow is sick
                    milk_income *= animal_constants.SICK_MILK_PRODUCTION_MULTIPLIER 
                    treatment_cost = animal_constants.TREATMENT_COST_PER_MONTH
                    if random.uniform(0, 1) < animal_constants.RECOVER_RATE:
                        next_disease = 0 # recovered from disease
                    else:
                        next_disease = 1 # remain sick
                else:
                    if random.uniform(0, 1) < animal_constants.DISEASE_RISK[parity]:
                        next_disease = 1 # become sick
                    else:
                        next_disease = 0 #remain healthy

                self.state = (next_parity, next_mac, next_mip, next_disease)
                reward = milk_income + calf_income - breed_cost - treatment_cost
                # print(">keep not died")
                # print("milk income:", milk_income, "calf_income:", calf_income, "breed_cost", breed_cost, "treatment_cost",treatment_cost)
        # print("one reward:", reward)
        return self.state, reward
            

    def render(self):
        print(f"Current state: {self.state}")

    def assign_woods_parameters(self, parity):
        if parity <= 3:
            self.parameter_a = animal_constants.WOODS_PARAMETERS[0][parity-1]
            self.parameter_b = animal_constants.WOODS_PARAMETERS[1][parity-1]
            self.parameter_c = animal_constants.WOODS_PARAMETERS[2][parity-1]
        else: 
            self.parameter_a = animal_constants.WOODS_PARAMETERS[0][-1]
            self.parameter_b = animal_constants.WOODS_PARAMETERS[1][-1]
            self.parameter_c = animal_constants.WOODS_PARAMETERS[2][-1]
        return self.parameter_a, self.parameter_b, self.parameter_c
    
    def get_y_values_wood_curve(self, t, parameter_a, parameter_b, parameter_c):
        return parameter_a * np.power(t, parameter_b) * np.exp(-1 * parameter_c * t)

    def calc_integral_wood_curve(self, t1, t2, parameter_a, parameter_b, parameter_c):
        result, _ = quad(self.get_y_values_wood_curve, t1, t2, args=(parameter_a, parameter_b, parameter_c))
        return result
    
    def breed(self, parity, mac, disease):
        random_num = random.uniform(0, 1)
        health_success_rate = max(0, animal_constants.PREG_RATE[parity] - (mac-3)*animal_constants.PREG_RATE_DROP)
        sick_success_rate = health_success_rate*animal_constants.SICK_PREG_RATE_MULTIPLIER 
        if disease == 0:
            return True if random_num < health_success_rate else False
        else: 
            return True if random_num < sick_success_rate else False 
    
    def slaughter(self, parity, disease): 
        if parity == 0 or parity == 1:
            bw = 0.82 * animal_constants.MANUTURE_BODY_WEIGHT
        elif parity == 2:
            bw = 0.92 * animal_constants.MANUTURE_BODY_WEIGHT
        else:
            bw = animal_constants.MANUTURE_BODY_WEIGHT
        return bw * animal_constants.SLAUGHTER_PRICE if disease == 0 else bw * animal_constants.SLAUGHTER_PRICE * animal_constants.SICK_SLAUGHTER_PRICE_MULTIPLIER 

    def death(self, parity, disease):
        random_num = random.uniform(0, 1)
        if disease == 0: #healthy
            if random_num < animal_constants.DEATH_RATE[parity]/100:
                return True
            else:
                return False
        else: 
            if random_num < animal_constants.SICK_DEATH_RATE_MULTIPLIER*animal_constants.DEATH_RATE[parity]/100:
                return True
            else:
                return False
