# MNIST
Handwritten digit recognition demo

# Requirement
- DISCO_F413ZH
- SD Card (see SD card section)

# Build Instruction
- clone the project
- run `mbed deploy`
- Copy from `sd_card` to the root of your SD card
- Insert it to DISCO_F413ZH
- Build commands:
```
mbed compile -m DISCO_F413ZH -t GCC_ARM --profile=utensor/build_profile/release.json -f
```

# Usage
- Wait for the LCD to initialize
- Draw a digit on the screen
- Press the User Button
- Press Rest to restart
