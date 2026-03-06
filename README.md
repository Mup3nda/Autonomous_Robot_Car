# Autonomous_Robot_Car

## Git Setup on Robot
Before committing, set your identity:
```bash
git config user.name "Your Name"
git config user.email "your@email.com"
git commit -m "your message"
```

This way, **nobody can commit without explicitly identifying themselves** first!

## After Making C++ Changes (cservoarm.cpp / cservoarm.h)

If you modify any C++ source files, you must rebuild the binary on the Raspberry Pi before the changes take effect.

1. SSH into the Raspberry Pi and pull the latest changes:
```bash
git pull
```

2. Kill the running teensy_interface binary:
```bash
pkill teensy_interfac
```

3. Navigate to the build folder and rebuild:
```bash
cd ~/svn/robobot/teensy_interface/build
cmake ..
make
```

> **Note:** You only need `cmake ..` if `CMakeLists.txt` changed. If only `.cpp`/`.h` files changed, `make` alone is enough.

4. Restart the binary:
```bash
./teensy_interface -d -l
```

> **Changes to `robot.ini` or `.py` files do NOT require a rebuild** — just `git pull` and restart the binary.
