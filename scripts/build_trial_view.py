from src.trial_view import build_trial_view, save_trial_view, PROCESSED_DIR

view = build_trial_view()
save_trial_view(view)

print(f"Saved to {PROCESSED_DIR}")
for name, df in view.items():
    print(f"  {name:9} rows={len(df):6}  columns={df.shape[1]}")