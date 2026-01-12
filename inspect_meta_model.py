import joblib

# Load the saved meta-model
meta = joblib.load("models/meta_model_aligned_multimodal.joblib")

print("=== Meta-model keys ===")
print(meta.keys())  # usually contains 'model' and 'features'

print("\n=== Features used ===")
print(meta["features"])  # which features went into stacking

print("\n=== Meta-model type ===")
print(type(meta["model"]))  # should be LogisticRegression or CalibratedClassifierCV
