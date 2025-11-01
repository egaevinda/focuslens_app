import h5py, json, numpy as np, shutil, os

SRC = r"D:\focuslens_app\model_final.h5"          
DST = r"D:\focuslens_app\model_final_patched.h5"   

assert os.path.exists(SRC), f"File tidak ditemukan: {SRC}"
shutil.copy2(SRC, DST)  

with h5py.File(DST, "r+") as f:
    
    raw = f.attrs.get("model_config", None)
    if raw is None:
        raise RuntimeError("Atribut 'model_config' tidak ditemukan di file H5 — tidak bisa dipatch.")

    cfg = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

    if '"batch_shape":' not in cfg and "'batch_shape':" not in cfg:
        print("Tidak ada 'batch_shape' di model_config — tidak perlu patch.")
    else:
        
        cfg2 = cfg.replace('"batch_shape":', '"batch_input_shape":').replace("'batch_shape':", "'batch_input_shape':")
        if cfg2 == cfg:
            raise RuntimeError("Gagal mengganti 'batch_shape' → 'batch_input_shape'.")

       
        f.attrs.modify("model_config", np.string_(cfg2))
        print("OK: 'batch_shape' diganti menjadi 'batch_input_shape' pada", DST)

print("Selesai. File PATCHED:", DST)
