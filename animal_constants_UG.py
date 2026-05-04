### Economic parameters for dairy cattle in 2025 ###
# Milk prices
MILK_PRICE = 22.6 # per cwt, average of 2024 Nov to 2025 Oct (12 months) from USDA uniform milk price https://mymarketnews.ams.usda.gov/filerepo/sites/default/files/3351/2025-11-01/1293713/ams_3351_00065.pdf 
# Feed cost
FEED_COST = 0.19 # unit: kg of DM (https://www.sciencedirect.com/science/article/pii/S0022030224007811#:~:text=Marginal%20Revenue%20and%20Cost,by%20the%20differing%20regression%20estimates)

# Replacement cost
REPLACEMENT_COST = 4135 # per hd, average of 2024 Nov to 2025 Oct (12 months) from USDA https://mymarketnews.ams.usda.gov/viewReport/2957
# Calf price
CALF_PRICE = 1160 # per hd, average of heifer calf and bull calf 2024 Nov to 2025 Oct (12 months) from USDA https://mymarketnews.ams.usda.gov/viewReport/2957
# Slaughter prices 
SLAUGHTER_PRICE = 3.78 # per kg, according to average of 2024 Nov to 2025 Oct (12 months) from USDA https://mymarketnews.ams.usda.gov/viewReport/2957 # avg of $266/cwt for dressed cow (breaker, boner, cutter) * 55% = $146.3/cwt = $146.3/45.3592 kg 3.23/kg
MANUTURE_BODY_WEIGHT = 740.1 # kg, according to Manfei repro paper: https://www.sciencedirect.com/science/article/pii/S0022030223001145#bib66
# Breeding cost
BREED_COST_PER_INSEM = 30 # estimated from 1) Manfei repro paper: https://www.sciencedirect.com/science/article/pii/S0022030223001145#bib66 ($15/straw, an arm service as $10/AI); 2)  https://doi.org/10.3389/fvets.2023.1345782: "The costs of reproduction programs are insemination costs (semen US$20/cow and labor US$5/cow) and ultrasound pregnancy monitoring (US$100/h)."

### Transitional probabilities and other parameters ###
# Reproduction
CONCEPTION_RATE = {1: 0.34, 2: 0.3, 3: 0.29, 4: 0.28, 5: 0.26, 6: 0.24, 7: 0.22, 8: 0.20, 9: 0.18, 10: 0.16, 11: 0.14, 12: 0.12} #https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1297&context=usdaarsfacpub&utm_source=chatgpt.com (Table 7), extrapolate for parity 7-12 by assuming a drop of 0.02 per parity after parity 6.
CONCEPTION_RATE_DROP = 0.02 # on average, drop 2% per insemination, https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1297&context=usdaarsfacpub&utm_source=chatgpt.com (Table 7)

# Death
DEATH_RATE = [0, 0.1725, 0.2244, 0.3154, 0.3725, 0.4117, 0.4949, 0.5072, 0.5496, 0.5940, 0.6386, 0.6834, 0.7285] #unit: %; https://www.sciencedirect.com/science/article/pii/S0022030208710865 #Table 2

# # General Disease
# DISEASE_RISK = [0, 0.15, 0.18, 0.2, 0.2, 0.23, 0.25, 0.28, 0.3, 0.3, 0.32, 0.35, 0.35] # from health to sick per month by parity
# # probabily too high for a monthly rate. https://pmc.ncbi.nlm.nih.gov/articles/PMC7114122/ is a potential source (table 5. about 86% incidence rate for all diseases per year, so ~7% per month)
# RECOVER_RATE = 0.6 # after treatment, 60% can recover (https://www.mdpi.com/2624-862X/2/4/45: only 10% to 40% recovered, no diff between treating or not treating)
# SICK_DEATH_RATE_MULTIPLIER = 2 #sick, twice of the normal dealth rate (https://www.journalofdairyscience.org/article/S0022-0302(11)00508-X/fulltext, table 5 and 6, but primiparous cows have a higher dealth rate compared to healthy cows, like 10: 1, multiparous has about 2:1)
# SICK_MILK_PRODUCTION_MULTIPLIER = 0.7 # general milk production drop about 30% (https://www.sciencedirect.com/science/article/pii/S0022030204731926#tbl4)
# SICK_SLAUGHTER_PRICE_MULTIPLIER = 0.9 # slaughter price 90% of healthy cow (https://pmc.ncbi.nlm.nih.gov/articles/PMC8281100/; https://www.sciencedirect.com/science/article/pii/S0022030218308075)
# SICK_CONCEPTION_RATE_MULTIPLIER = 0.9 #https://www.sciencedirect.com/science/article/pii/S002203022400821X, ~0.26% to 14.67% increase in calving interval depends on disease (Table 4)
# # Disease treatment cost
# TREATMENT_COST_PER_MONTH = 98.9 # average of 7 diseases from https://www.sciencedirect.com/science/article/pii/S0022030216308992#tbl8

# Disease - Mastitis
MASTITIS_TREATMENT_COST_PER_MONTH = 78 # https://www.sciencedirect.com/science/article/pii/S0022030216308992#tbl8
MASTITIS_DISEASE_RISK = [0, 0.02, 0.027, 0.045, 0.0495, 0.0545, 0.0600, 0.0660, 0.0726, 0.0799, 0.0879, 0.0967, 0.1064] # https://www.sciencedirect.com/science/article/pii/S0022030294772398 -> convert from "the incidence rates of
                                                                                                        # clinical mastitis in the period from 1 wk before
                                                                                                        # calving until 10 mo after calving were 6.6,9.0,
                                                                                                        # and 14.7 cases per 10,OOO cow days at risk for
                                                                                                        # first, second, and third lactations, respectively"
                                                                                                        # Conversion:
                                                                                                        # 10,000 cow-days ≈ 328 cow-months (10,000 / 30.5)
                                                                                                        # 6.6/328 = 0.0201
                                                                                                        # 9.0/328 = 0.0274
                                                                                                        # 14.7/328 = 0.0448
                                                                                                        # So the monthly risk is about 2.0%, 2.7%, and 4.5% for parity 1, 2, and 3+, respectively.
MASTITIS_RECOVER_RATE = 0.65 # https://pmc.ncbi.nlm.nih.gov/articles/PMC8636650/#:~:text=At%20the%20end%20of%20the,14%20and%2021%20post%2Dtreatment. https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2023.1079269/full 
MASTITIS_SICK_DEATH_RATE_MULTIPLIER = 2.3 #sick, twice of the normal dealth rate (https://www.journalofdairyscience.org/article/S0022-0302(11)00508-X/fulltext, table 5 and 6, but primiparous cows have a higher dealth rate compared to healthy cows, like 10: 1, multiparous has about 2:1)
MASTITIS_SICK_MILK_PRODUCTION_MULTIPLIER = 0.46 # https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2022.1070051/full: "The average daily milk yield of cows with clinical mastitis (Mean ± SEM; 18.6 ± 0.54 kg) was significantly (p < 0.001) lower than the average daily milk yield of clinical mastitis free cows (40.5 ± 0.29 kg). "
MASTITIS_SICK_SLAUGHTER_PRICE_MULTIPLIER = 0.95 # slight decrease (https://pmc.ncbi.nlm.nih.gov/articles/PMC8281100/; https://www.sciencedirect.com/science/article/pii/S0022030218308075)
MASTITIS_SICK_CONCEPTION_RATE_MULTIPLIER = 0.68 #https://www.sciencedirect.com/science/article/pii/S002203022400821X, (Table 4): Fertility impact of CM: +8.42% CI
                                                # assume a 400 days calving interval for healthy cows, then the conception rate drop is about 4% (400 days vs 433.7 days) for mastitis cows
                                                # assume Baseline conception rate = 0.3 and estus cycle = 21 days
                                                # solve eqn: 21(1/cr_mas -1) = 21(1/0.3 -1)
                                                # we get cr_mas = 0.203
                                                # multiplier = 0.203/0.3 = 0.677

# Lactation curve
WOODS_PARAMETERS = [[15.72, 22.06, 21.92], [0.2433, 0.235, 0.2627], [0.002445, 0.003642, 0.004041]] # [a], [b], and [c], each list have 3 values for parity 1, 2, and 3+, based on Manfei 2022 paper. Mean + parity adjustment
MILK_PRODUCTION_DISCOUNT_FACTOR_FROM_P3_DICTIONARY={4: 1.035, 5: 1.049, 6: 1.046, 7: 1.036, 8: 1.018, 9: 0.995, 10: 0.957, 11: 0.918, 12: 0.88} # same lactation curve shape as P3+ above, but different height (https://www.sciencedirect.com/science/article/pii/S2666910225000924?via%3Dihub#tbl1fn1) they have P1 to P10, P11 and P12 are estimated by extrapolation. 
