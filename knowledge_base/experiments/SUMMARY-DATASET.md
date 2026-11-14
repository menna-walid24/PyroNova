---
experiment_id: SUMMARY-DATASET
doc_type: summary
source_investigation: PSI-159 (CFI-G)
---

# Dataset overview — NASA PSI-159 CFI-G Normal Flames

## Summary

The knowledge base contains 179 real tests from the NASA PSI-159 CFI-G (Cool Flames
Investigation with Gases) experiment on the International Space Station (microgravity), run on
11 test days from 2021-03-29 to 2021-10-18.
Each test burns a small gaseous fuel jet (ethane, propane or butane, pure or diluted with
nitrogen) in an oxygen/nitrogen atmosphere.

- Oxygen: 18.0-40.0%. There are NO tests between 23.5% and 33% oxygen.
- Pressure: 51-304 kPa.
- Fuel flow: 0.0-3.215 mg/s.
- Outcomes: 132 of 179 tests ignited a hot flame; 102 self-extinguished
  and 30 were still burning when the crew cut the fuel flow
  (47 did not ignite).
- Median burn time of lit flames 35.0 s, longest 304.0 s.
- Recorded fields: hot_ignition, burn_time_s, end_mode (self_extinguished / flow_terminated / none),
  peak_burner_temp_c, cool_flame, plus a hazard index (0-100) computed from the outcome.

Note: The 33-40% oxygen levels are ISS test-chamber conditions, not a spacecraft cabin atmosphere (ISS cabin air is about 21%).
