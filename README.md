- Install
```
git clone https://github.com/tangefly/HGEMM.git
pip install -e .
```

- Benchmark

```
python ./benchmarks/xxx_bench.py
```

- Test

```
pytest -v
```

- Profile

```
mkdir -p results
# full
ncu \
  --set full \
  --kernel-name naive_hgemm_kernel \
  -o ./results/naive_hgemm_kernel_profile \
  python ./profiles/naive_hgemm_profile.py

# roofline
ncu \
  --kernel-name naive_hgemm_kernel \
  --section SpeedOfLight_HierarchicalHalfRooflineChart \
  -o ./results/naive_hgemm_kernel_roofline
```