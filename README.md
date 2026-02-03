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
ncu \
  --set full \
  --kernel-name naive_hgemm_kernel \
  -o ./results/naive_hgemm_kernel_profile \
  python ./profiles/naive_hgemm_profile.py

ncu \
  --set full \
  --kernel-name hierarchical_tiling_hgemm_kernel \
  -o ./results/hierarchical_tiling_hgemm_kernel_profile \
  python ./profiles/hierarchical_tiling_hgemm_profile.py
```