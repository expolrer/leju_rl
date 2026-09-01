# Kuavo S52 stairs teacher

`model_92099.pt` is the protected S53 teacher used to warm-start and distill the
S52 policy. It is not an accepted S52 stair policy and must not be deployed as
one without S52-specific Isaac Lab and MuJoCo validation.

SHA-256:

```text
6483ff66456f1e218713f228114a89b6bd5688d94d4c2612ebc247375812f0f4
```

Validated interfaces:

- policy joints: 27;
- simulator joints: 29, including two head joints;
- policy observations: 148;
- actions: 27.

The nominal S52 stand/walk checks pass in Isaac Lab and MuJoCo. Full stair
validation remains pending.
