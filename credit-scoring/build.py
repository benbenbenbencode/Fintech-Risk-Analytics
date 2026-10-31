"""Build everything headlessly:  python build.py [--customers 100000]"""
import argparse
import time

from studio import cannibal, data, generate_world, models

ap = argparse.ArgumentParser()
ap.add_argument("--customers", type=int, default=100_000)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
t = time.time()
print(generate_world.generate(a.customers, seed=a.seed, out_dir=str(data.WORLD)))
models.build_all()
cannibal.analyse(data.WORLD)
print(f"done in {time.time() - t:.0f}s -> streamlit run app.py")
