# Fluent journal template

Copy `run_case.jou.template` to work_dir, set boundary conditions from `design_params.json`, run:

```bash
fluent 3ddp -g < run_case.jou
echo "cd=0.251" > result_cd.txt
```
