# change state of MIM to MAC (month after calving)
import numpy as np
from scipy.integrate import quad
import random
import utility
# Dynamic import based on scenario - modified to support command-line selection
# Default scenario is '2025' if not specified
_scenario = '2025'  # Default scenario
animal_constants = None  # Will be set by set_scenario()

def set_scenario(scenario='2025'):
    """Set the animal constants scenario dynamically.
    
    Args:
        scenario (str): One of '2025', 'OG', 'OB', 'UG', 'UB'
    """
    global animal_constants, _scenario
    _scenario = scenario
    
    scenario_map = {
        '2025': 'animal_constants_2025',
        'OG': 'animal_constants_OG',
        'OB': 'animal_constants_OB',
        'UG': 'animal_constants_UG',
        'UB': 'animal_constants_UB'
    }
    
    if scenario not in scenario_map:
        raise ValueError(f"Unknown scenario: {scenario}. Must be one of {list(scenario_map.keys())}")
    
    module_name = scenario_map[scenario]
    animal_constants = __import__(module_name)
    print(f"Loaded scenario: {scenario} (module: {module_name})")
    return animal_constants

# Initialize with default scenario
set_scenario(_scenario)

class CowEnv:
    def __init__(self, parity_range, mac_range, mip_range, disease_range):
        """Initialize the environment with allowed state ranges and default state.

        Args:
            parity_range (Sequence[int]): Valid parities that can appear in the state.
            mac_range (Sequence[int]): Allowed months-after-calving values.
            mip_range (Sequence[int]): Allowed months-in-pregnancy values.
            disease_range (Sequence[int]): Allowed disease status flags (typically 0/1).
        """
        self.state = (0, 0, 9, 0)
        self.parameter_a = self.parameter_b = self.parameter_c = None
        self.actions = ['keep', 'replace']
        self.parity_range = parity_range
        self.mac_range = mac_range
        self.mip_range = mip_range
        self.disease_range = disease_range

    def reset(self):
        """Sample a random valid state with realistic disease prevalence."""
        # Sample parity, MAC, MIP uniformly
        parity = random.choice(self.parity_range)
        mac = random.choice(self.mac_range)
        mip = random.choice(self.mip_range)
        
        # FIXED: Sample disease with realistic prevalence
        incidence = animal_constants.MASTITIS_DISEASE_RISK[parity]
        recovery = animal_constants.MASTITIS_RECOVER_RATE
        disease_prevalence = incidence / (incidence + recovery) if (incidence + recovery) > 0 else 0
        disease = 1 if random.random() < disease_prevalence else 0
        
        self.state = (parity, mac, mip, disease)
        
        # Ensure valid state
        while not utility.possible_state2(self.state, self.parity_range, self.mac_range, 
                                        self.mip_range, self.disease_range):
            parity = random.choice(self.parity_range)
            mac = random.choice(self.mac_range)
            mip = random.choice(self.mip_range)
            incidence = animal_constants.MASTITIS_DISEASE_RISK[parity]
            recovery = animal_constants.MASTITIS_RECOVER_RATE
            disease_prevalence = incidence / (incidence + recovery) if (incidence + recovery) > 0 else 0
            disease = 1 if random.random() < disease_prevalence else 0
            self.state = (parity, mac, mip, disease)
        
        return self.state

    def step(self, action):
        """Advance the environment one month based on the action.

        Args:
            action (str): Either 'keep' or 'replace'.

        Returns:
            tuple[tuple[int, int, int, int], float]: Next state and scalar reward.
        """
        slaughter_income = 0
        calf_income = 0
        milk_income = 0
        breed_cost = 0
        treatment_cost = 0
        feed_cost = 0
        parity, mac, mip, disease = self.state
        # print("state:", parity, mac, mip, disease)

        if action == 'replace':
            slaughter_income = self.calculate_slaughter_income(parity, disease) 
            self.state = (0, 0, 9, 0) # replaced by a new springer
            feed_cost = self.calculate_feed_cost(self.state[0], self.state[1], self.state[2], self.state[3])
            reward = slaughter_income - animal_constants.REPLACEMENT_COST - feed_cost
            # print(">replace")
            # print("slaughter_income:", slaughter_income, "replacement_cost", animal_constants.REPLACEMENT_COST )
        else: # 'keep' 
            next_parity = parity # by default, parity does not change, unless mip == 9
            next_mac = 0
            next_mip = 0
            next_mip = 0
            next_disease = 0
            # death
            if self.death_status(parity, disease): # died
                # print(">keep died")
                slaughter_income = 0 # it's 0 because it is a dead cow
                self.state = (0, 0, 9, 0) # replaced by a new springer
                feed_cost = self.calculate_feed_cost(self.state[0], self.state[1], self.state[2], self.state[3])
                reward = slaughter_income - animal_constants.REPLACEMENT_COST - feed_cost
                # print("slaughter_income:", slaughter_income, "replacement_cost", animal_constants.REPLACEMENT_COST )
            else:
                # milking
                milk_income = self.calculate_milk_income(parity, mac, mip, disease)
                next_mac = mac + 1

                # pregnancy status
                if mip == 9: # calving
                    calf_income = animal_constants.CALF_PRICE
                    next_parity = parity+1
                    next_mac = 1
                    next_mip = 0
                elif mip == 0: # breeding
                    if mac>=3:
                        breed_cost = animal_constants.BREED_COST_PER_INSEM  
                        if self.breed_success(parity, mac, disease) == True:
                            next_mip = 1
                        else: 
                            next_mip = 0
                else: # keep pregnancy
                    next_mip = mip + 1

                # Disease affect slaughter income (in calculate_slaughter_income() function), milk income (in calculate_milk_production() function), breed success (in breed() function), and treatment_cost 
                if disease == 1: # when the cow is sick
                    treatment_cost = animal_constants.MASTITIS_TREATMENT_COST_PER_MONTH
                    if random.uniform(0, 1) < animal_constants.MASTITIS_RECOVER_RATE:
                        next_disease = 0 # recovered from disease
                    else:
                        next_disease = 1 # remain sick
                else:
                    if random.uniform(0, 1) < animal_constants.MASTITIS_DISEASE_RISK[parity]:
                        next_disease = 1 # become sick
                    else:
                        next_disease = 0 #remain healthy

                self.state = (next_parity, next_mac, next_mip, next_disease)
                feed_cost = self.calculate_feed_cost(self.state[0], self.state[1], self.state[2], self.state[3])
                reward = milk_income + calf_income - breed_cost - treatment_cost - feed_cost
                # print(">keep not died")
                # print("milk income:", milk_income, "calf_income:", calf_income, "breed_cost", breed_cost, "treatment_cost",treatment_cost)
        # print("one reward:", reward)
        return self.state, reward
            

    def render(self):
        """Print the current state tuple for quick inspection.

        Returns:
            None
        """
        print(f"Current state: {self.state}")

    def assign_woods_parameters(self, parity):
        """Return Woods curve parameters appropriate for the current parity.

        Args:
            parity (int): Current parity of the cow.

        Returns:
            tuple[float, float, float]: Parameters (a, b, c) for the Woods curve.
        """
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
        """Compute the Woods curve estimated milk production (kg/d) at time t for supplied parameters.

        Args:
            t (float): Day in milk.
            parameter_a (float): Woods parameter a.
            parameter_b (float): Woods parameter b.
            parameter_c (float): Woods parameter c.

        Returns:
            float: Milk production value (kg/d) at day t.
        """
        return parameter_a * np.power(t, parameter_b) * np.exp(-1 * parameter_c * t)

    def calc_integral_wood_curve(self, t1, t2, parameter_a, parameter_b, parameter_c):
        """Integrate the Woods curve between days t1 and t2 to estimate monthly milk.

        Args:
            t1 (float): Start day of integration window.
            t2 (float): End day of integration window.
            parameter_a (float): Woods parameter a.
            parameter_b (float): Woods parameter b.
            parameter_c (float): Woods parameter c.

        Returns:
            float: Integrated production between t1 and t2.
        """
        result, _ = quad(self.get_y_values_wood_curve, t1, t2, args=(parameter_a, parameter_b, parameter_c))
        return result

    def calculate_monthly_milk_production(self, parity, mac, mip, disease):
        """Estimate monthly milk production using Woods curve. Discounts for disease if applicable.

        Args:
            parity (int): Cow parity.
            mac (int): Month after calving.
            mip (int): Month in pregnancy.
            disease (int): Disease indicator (0 healthy, 1 sick).
        Returns:
            float: Estimated monthly milk production (kg/month).
        """
        if mac == 0 or mip == 7 or mip == 8 or mip == 9: # springer or dry cow
            return 0
        self.parameter_a, self.parameter_b, self.parameter_c = self.assign_woods_parameters(parity)
        dim = (mac-1)*30 + 1
        milk_production = self.calc_integral_wood_curve(dim, dim+30, self.parameter_a, self.parameter_b, self.parameter_c)
        if disease == 1:
            milk_production *= animal_constants.MASTITIS_SICK_MILK_PRODUCTION_MULTIPLIER
        return milk_production
    
    def calculate_milk_income(self, parity, mac, mip, disease):
        """Calculate milk income for given parity, MAC, and disease status.

        Args:
            parity (int): Cow parity.
            mac (int): Month after calving.
        Returns:
            float: Estimated milk income for the month.
        """
        milk_production = self.calculate_monthly_milk_production(parity, mac, mip, disease)
        milk_income = milk_production*2.2/100 * animal_constants.MILK_PRICE  
        return milk_income
    
    def _estimate_daily_dmi(self, parity, mac, mip, disease):
        """Return daily DMI (kg/d) using milk, body weight, and weeks in milk."""
        if parity == 0:  # heifer springer 
            return 9 # hard-coded fixed 9 kg/day for springer heifer

        monthly_milk = self.calculate_monthly_milk_production(parity, mac, mip, disease)
        avg_milk_per_day = monthly_milk / 30.0
        metabolic_bw = self.get_body_weight(parity) ** 0.75
        dim_start = (mac - 1) * 30 + 1
        dim_end = mac * 30
        avg_dim = (dim_start + dim_end) / 2.0
        weeks_in_milk = avg_dim / 7.0
        return (0.372 * avg_milk_per_day + 0.0968 * metabolic_bw) * (1 - np.exp(-0.192 * (weeks_in_milk + 3.67)))
    
    def get_monthly_dmi(self, parity, mac, mip, disease):
        """Estimate monthly dry matter intake (kg/month) for the given state."""
        return self._estimate_daily_dmi(parity, mac, mip, disease) * 30.0
    
    def calculate_feed_cost(self, parity, mac, mip, disease):
        """Estimate monthly feed cost based on DMI and feed price.

        Args:
            parity (int): Cow parity.
            mac (int): Month after calving.
            mip (int): Month in pregnancy.
            disease (int): Disease indicator (0 healthy, 1 sick).  
        Returns:
            float: Estimated monthly feed cost.
        """
        monthly_dmi = self.get_monthly_dmi(parity, mac, mip, disease)
        feed_cost = monthly_dmi * animal_constants.FEED_COST
        return feed_cost
    
    def breed_success(self, parity, mac, disease):
        """Stochastically determine whether breeding succeeds given parity, MAC, and disease.

        Args:
            parity (int): Current parity of the cow.
            mac (int): Month after calving.
            disease (int): Disease indicator (0 healthy, 1 sick).

        Returns:
            bool: True if breeding succeeds, False otherwise.
        """
        random_num = random.uniform(0, 1)
        health_success_rate = max(0, animal_constants.CONCEPTION_RATE[parity] - (mac-3)*animal_constants.CONCEPTION_RATE_DROP)
        sick_success_rate = health_success_rate*animal_constants.MASTITIS_SICK_CONCEPTION_RATE_MULTIPLIER 
        if disease == 0:
            return True if random_num < health_success_rate else False
        else: 
            return True if random_num < sick_success_rate else False 
    
    def get_body_weight(self, parity):
        """Estimate body weight based on parity.

        Args:
            parity (int): Cow parity.

        Returns:
            float: Estimated body weight in kg.
        """
        if parity == 0 or parity == 1:
            bw = 0.82 * animal_constants.MANUTURE_BODY_WEIGHT
        elif parity == 2:
            bw = 0.92 * animal_constants.MANUTURE_BODY_WEIGHT
        else:
            bw = animal_constants.MANUTURE_BODY_WEIGHT
        return bw
    
    def calculate_slaughter_income(self, parity, disease): 
        """Compute slaughter value based on parity-specific body weight and disease status.

        Args:
            parity (int): Cow parity.
            disease (int): Disease indicator (0 healthy, 1 sick).

        Returns:
            float: Expected slaughter income for the cow.
        """
        bw = self.get_body_weight(parity)
        return bw * animal_constants.SLAUGHTER_PRICE if disease == 0 else bw * animal_constants.SLAUGHTER_PRICE * animal_constants.MASTITIS_SICK_SLAUGHTER_PRICE_MULTIPLIER 

    def death_status(self, parity, disease):
        """Randomly determine whether the cow dies this month based on parity and disease.

        Args:
            parity (int): Cow parity.
            disease (int): Disease indicator (0 healthy, 1 sick).

        Returns:
            bool: True if the cow dies this month, otherwise False.
        """
        random_num = random.uniform(0, 1)
        if disease == 0: #healthy
            if random_num < animal_constants.DEATH_RATE[parity]/100:
                return True
            else:
                return False
        else: 
            if random_num < animal_constants.MASTITIS_SICK_DEATH_RATE_MULTIPLIER*animal_constants.DEATH_RATE[parity]/100:
                return True
            else:
                return False
    
