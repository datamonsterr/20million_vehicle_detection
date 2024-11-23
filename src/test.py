import pickle
def update():
    print("a")

secret = {}

secret["update"] = update
with open("a.pkl", "wb") as f:
    pickle.dump(secret, f)

with open("a.pkl", "rb") as file:
    loaded_data = pickle.load(file)  # Load the serialized object

loaded_data["update"]()