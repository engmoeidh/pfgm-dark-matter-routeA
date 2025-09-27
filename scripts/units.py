
import yaml, os

_DEFAULT = {
    "speed_of_light_km_s": 299_792.458,
    "G_kpc_km2_s2_Msun": 4.30091e-6
}

def load_units(path="units.yaml"):
    if os.path.exists(path):
        with open(path,"r") as f:
            u = yaml.safe_load(f)
    else:
        u = dict(_DEFAULT)
    for k,v in _DEFAULT.items():
        if k not in u: u[k] = v
    return u
