"""Lightweight scenario checks for `CowEnv` feed and milk calculations."""

from dataclasses import dataclass

import cow_environment2


@dataclass(frozen=True)
class Scenario:
	label: str
	parity: int
	mac: int
	mip: int
	disease: int


def summarize(env: cow_environment2.CowEnv, scenario: Scenario) -> None:
	"""Print per-day milk, DMI, and feed cost for a scenario."""
	state = (scenario.parity, scenario.mac, scenario.mip, scenario.disease)
	monthly_milk = env.calculate_monthly_milk_production(*state)
	monthly_dmi = env.get_monthly_dmi(*state)
	monthly_feed = env.calculate_feed_cost(*state)

	print(f"[{scenario.label}] state={state}")
	print(f"  milk/day      : {monthly_milk / 30:.2f} kg")
	print(f"  dmi/day       : {monthly_dmi / 30:.2f} kg")
	print(f"  feed cost/day : ${monthly_feed / 30:.2f}")
	print()


def main() -> None:
	env = cow_environment2.CowEnv(
		parity_range=range(0, 6),
		mac_range=range(0, 13),
		mip_range=range(0, 10),
		disease_range=[0, 1],
	)

	scenarios = [
		Scenario("Springer heifer", 0, 0, 9, 0),
		Scenario("Fresh parity 2", 2, 1, 0, 0),
		Scenario("Peak parity 2", 2, 3, 0, 0),
		Scenario("Late parity 2", 2, 10, 3, 0),
		Scenario("Fresh parity 3", 3, 2, 0, 0),
		Scenario("Peak parity 3", 3, 3, 0, 0),
		Scenario("Late parity 3", 3, 10, 4, 0),
		Scenario("Late parity 3 (sick)", 3, 10, 4, 1),
		Scenario("Dry parity 3 (mip=7)", 3, 10, 7, 0),
		Scenario("Dry parity 3 (mip=8)", 3, 10, 8, 0),
		Scenario("Calving parity 3", 3, 10, 9, 0),
	]

	for scenario in scenarios:
		summarize(env, scenario)


if __name__ == "__main__":
	main()