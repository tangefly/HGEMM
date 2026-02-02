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
```

/Applications/NVIDIA\ Nsight\ Compute.app/Contents/MacOS/ncu-ui \
  naive_hgemm_kernel_profile.ncu-rep
