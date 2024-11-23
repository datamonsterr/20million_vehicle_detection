import pickle

a = {}
a['need_enhanced_image'] = 'src_2_frame'
with open("secret.pkl", "wb") as f:
    pickle.dump(a, f)