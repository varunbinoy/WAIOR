# WAIOR — MBA Timetable Optimization System

AI + Operations Research based MBA timetable optimization using CP-SAT.

## Objective
- 10-week term
- 20 sessions per course
- 10 rooms (Weeks 1–4) → 4 rooms (Weeks 5–10)
- Zero student conflict
- Zero faculty conflict
- Section cap ≤ 70

## Key Results
- Theoretical max sessions: 900
- Fair solution achieved: 886
- Remaining deficit: 54
- Minimum contingent days required: 6

Conflict density — not room capacity — was binding.

## Where is the Final Schedule?

Core 10-week optimized timetable:
outputs/term_schedule_floor18.csv

Section allocation:
outputs/sections.csv
outputs/section_enrollments.csv

Contingent recovery schedule:
outputs/contingent_schedule.csv

## How to Run
python src/01_load_data.py
python src/02c_sectioning_ilp.py
python src/03_build_slots.py
python src/44_floor18_cap20_max_facsplit.py
python src/61_solve_contingent_days_min.py

Author: Varun Binoy (IIM Ranchi)
Generated: 2026-02-23 14:38