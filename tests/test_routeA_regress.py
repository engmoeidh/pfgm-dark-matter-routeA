import os, json

def test_binA_fit_outputs_exist():
    for p in ["results/tables/binA_fit_freeA.json", "results/tables/binA_fit_A1.json"]:
        assert os.path.isfile(p), f"Missing fit output: {p}"
        with open(p,"r") as f:
            d = json.load(f)
        assert "chi2" in d and "lam" in d
