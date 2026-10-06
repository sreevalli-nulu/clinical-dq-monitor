# Assumptions used across the project. Real studies set these in a monitoring plan,
# so we keep them in one place where they are easy to change.
VISIT_WINDOW_DAYS = 7      # allowed deviation from the planned visit day
WEIGHT_JUMP_PCT = 15       # weight change between visits that triggers a flag
AE_PRE_DOSE_DAYS = 30      # AE starting this many days before first dose is suspicious
MIN_SITE_SUBJECTS = 5      # sites with fewer dosed subjects are too small for statistics