### Economic parameters for dairy cattle in 2025 ###
# Milk prices
MILK_PRICE = 19.9 # per cwt, average of 2024 Nov to 2025 Oct (12 months) from USDA https://mymarketnews.ams.usda.gov/viewReport/2957 TODO: wait Jack to send me the mailbox values
# Replacement cost
REPLACEMENT_COST = 2757 # per hd, average of 2024 Nov to 2025 Oct (12 months) from USDA https://mymarketnews.ams.usda.gov/viewReport/2957
# Calf price
CALF_PRICE = 773 # per hd, average of heifer calf and bull calf 2024 Nov to 2025 Oct (12 months) from USDA https://mymarketnews.ams.usda.gov/viewReport/2957
# Slaughter prices 
SLAUGHTER_PRICE = 3.15 # per kg, according to average of 2024 Nov to 2025 Oct (12 months) from USDA https://mymarketnews.ams.usda.gov/viewReport/2957 # avg of $266/cwt for dressed cow (breaker, boner, cutter) * 60% = $159.88/cwt = 3.15/kg
MANUTURE_BODY_WEIGHT = 740.1 # kg, according to Manfei repro paper: https://www.sciencedirect.com/science/article/pii/S0022030223001145#bib66
# Breeding cost
BREED_COST_PER_INSEM = 30 # estimated from 1) Manfei repro paper: https://www.sciencedirect.com/science/article/pii/S0022030223001145#bib66 ($15/straw, an arm service as $10/AI); 2)  https://doi.org/10.3389/fvets.2023.1345782: "The costs of reproduction programs are insemination costs (semen US$20/cow and labor US$5/cow) and ultrasound pregnancy monitoring (US$100/h)."
# Disease treatment cost
TREATMENT_COST_PER_MONTH = 98.9 # average of 7 diseases from https://www.sciencedirect.com/science/article/pii/S0022030216308992#tbl8

### Transitional probabilities and other parameters ###
# Reproduction
CONCEPTION_RATE = {1: 0.34, 2: 0.3, 3: 0.29, 4: 0.28, 5: 0.26, 6: 0.24, 7: 0.24, 8: 0.24, 9: 0.24, 10: 0.24, 11: 0.24, 12: 0.24} #https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1297&context=usdaarsfacpub&utm_source=chatgpt.com (Table 7)
CONCEPTION_RATE_DROP = 0.02 # on average, drop 2% per insemination, https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1297&context=usdaarsfacpub&utm_source=chatgpt.com (Table 7)
SICK_CONCEPTION_RATE_MULTIPLIER = 0.9 #https://www.sciencedirect.com/science/article/pii/S002203022400821X, ~0.26% to 14.67% increase in calving interval depends on disease (Table 4)

# Death
DEATH_RATE = [0, 2.05, 2.66, 3.72, 4.38, 4.83, 5.78, 5.92, 6.40, 6.40, 6.40, 6.40, 6.40] #unit: %; https://www.sciencedirect.com/science/article/pii/S0022030208710865 #Table 2

# Disease
DISEASE_RISK = [0, 0.15, 0.18, 0.2, 0.2, 0.23, 0.25, 0.28, 0.3, 0.3, 0.32, 0.35, 0.35] # from health to sick per month by parity
# probabily too high for a monthly rate. https://pmc.ncbi.nlm.nih.gov/articles/PMC7114122/ is a potential source (table 5. about 86% incidence rate for all diseases per year, so ~7% per month)
RECOVER_RATE = 0.6 # after treatment, 60% can recover (https://www.mdpi.com/2624-862X/2/4/45: only 10% to 40% recovered, no diff between treating or not treating)
SICK_DEATH_RATE_MULTIPLIER = 2 #sick, twice of the normal dealth rate (https://www.journalofdairyscience.org/article/S0022-0302(11)00508-X/fulltext, table 5 and 6, but primiparous cows have a higher dealth rate compared to healthy cows, like 10: 1, multiparous has about 2:1)
SICK_MILK_PRODUCTION_MULTIPLIER = 0.7 # milk production (https://www.sciencedirect.com/science/article/pii/S0022030204731926#tbl4)
SICK_SLAUGHTER_PRICE_MULTIPLIER = 0.9 # slaughter price 90% of healthy cow (https://pmc.ncbi.nlm.nih.gov/articles/PMC8281100/; https://www.sciencedirect.com/science/article/pii/S0022030218308075)

# Lactation curve
WOODS_PARAMETERS = [[15.72, 22.06, 21.92], [0.2433, 0.235, 0.2627], [0.002445, 0.003642, 0.004041]] # [a], [b], and [c], each list have 3 values for 3 parity, based on Manfei 2022 paper. Mean + parity adjustment

### For later use
# if we want to differentiate bull calf and heifer calf price (we can also keep heifer and calculate a shadow price for them)
# MALE_CALF_PRICE = 804 # per hd, average of 2024 Nov to 2025 Oct (12 months) from USDA https://mymarketnews.ams.usda.gov/viewReport/2957
# FEMALE_CALF_PRICE = 742 # per hd, average of 2024 Nov to 2025 Oct (12 months) from USDA https://mymarketnews.ams.usda.gov/viewReport/2957

# Feed
# https://www.ers.usda.gov/data-products/milk-cost-of-production-estimates shows feed cost by state and size of operation (lastest: 2021- 2024)
# FEED_COST = 0.228 # unit: kg/milk ### original figure: 10.32 dollars per cwt milk sold, US, 2024 -> useless because it is whole-farm average, not cow-level. It mixes together feed consumed by lactating cows + dry cows + heifers + calves, and divides by total milk
# FEED_COST = 0.24 # unit: kg of DM (Manfei 2023 repro paper: https://www.sciencedirect.com/science/article/pii/S0022030223001145#bib66)
