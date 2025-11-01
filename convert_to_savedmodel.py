import keras, tensorflow as tf, os
SRC = r"D:\focuslens_app\model_final.h5"     
DST = r"D:\focuslens_app\model_sm"           

if not os.path.exists(SRC):
    raise FileNotFoundError(SRC)

model = keras.saving.load_model(SRC, compile=False, safe_mode=False)  
tf.saved_model.save(model, DST)                                       
print("SavedModel exported to:", DST)
