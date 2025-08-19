import pickle

# Assuming 'my_scaler_object' is the object you want to save
with open("results/p100o9_raw_hybrid/scaler.pkl", "wb") as f:  # 'wb' for binary write mode
    pickle.dump(my_scaler_object, f)
