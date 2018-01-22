# MNIST
Handwritten digit recognition demo

# Requirement
- DISCO_F413ZH
- SD Card (see SD card section)

# Build Instruction
- import the project:
`mbed import https://github.com/ARMmbed/mnist`
- Loading the SD card
- Copy from `utensor/TESTS/scripts/PRE-GEN/deep-mlp` to `/fs/testData/deep-mlp`
- Insert it to DISCO_F413ZH
- Build commands:
```
mbed compile -m DISCO_F413ZH -t GCC_ARM --profile=utensor/build_profile/release.json -f
```

# Expected Outputs
Serial Baud Rate: 115200
```
program start...
Inferencing...
PASSED 0.00000000

prediction: 7
```
