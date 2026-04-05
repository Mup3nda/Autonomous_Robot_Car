#!/bin/bash
# script to start applications after a reboot

TRACE_LOG=/home/local/on_reboot_trace.log
echo "$(date '+%F %T') REPO on_reboot.bash started" >> "$TRACE_LOG"

# Run the app to show Raspberry's IP on the Teensy display.
mkdir -p /home/local/Autonomous_Robot_Car/svn/log
cd /home/local/Autonomous_Robot_Car/svn/log
# save the last reboot date
echo "================ Rebooted ================" >> rebootinfo.txt
date >> rebootinfo.txt
../robobot/ip_disp/build/ip_disp 2>/dev/null >>ip_disp.out &
# save PID for debugging
echo "ip_disp started with PID:" >> rebootinfo.txt
sleep 0.1
pgrep -l ip_disp >> rebootinfo.txt

#
# start camera servers (allow cameras to be detected)
sleep 0.2
cd /home/local/Autonomous_Robot_Car/svn/robobot/stream_server
/usr/bin/python3 -u usb_stream_server.py 2>usb_stream_server.err >usb_stream_server.out & USB_PID=$!
echo "python3 usb_cam streamer started with PID: $USB_PID" >> /home/local/Autonomous_Robot_Car/svn/log/rebootinfo.txt
sleep 0.1
# /usr/bin/python3 stream_server.py 2>stream_server.err >stream_server.out & RASPI_CAM_PID=$!
# echo "python3 cam streamer started with PID: $RASPI_CAM_PID" >> /home/local/Autonomous_Robot_Car/svn/log/rebootinfo.txt
# sleep 0.1
pgrep -l python >> /home/local/Autonomous_Robot_Car/svn/log/rebootinfo.txt

#
# start teensy_interface - allow Teensy to be detected and interface loaded
# start is postponed until date and time is updated (typically after 25sec)
sleep 0.5
cd /home/local/Autonomous_Robot_Car/svn/robobot/teensy_interface

# Pull latest code from git
git fetch
CHANGES=$(git diff HEAD origin/main --name-only | grep -E '\.(cpp|h)$|CMakeLists.txt')
git pull

# Rebuild if binary missing OR if C++ files changed
if [ ! -f "build/teensy_interface" ] || [ -n "$CHANGES" ]; then
    echo "Rebuilding teensy_interface..." >> /home/local/Autonomous_Robot_Car/svn/log/rebootinfo.txt
    mkdir -p build
    cd build
    cmake ..
    make
    cd ..
fi
cd /home/local/Autonomous_Robot_Car/svn/robobot/teensy_interface
./build/teensy_interface -l 2>build/out_err.txt >build/out_console.txt &
echo "Teensy_interface started with PID:" >> /home/local/svn/log/rebootinfo.txt
sleep 0.1
pgrep -l teensy_i >> /home/local/Autonomous_Robot_Car/svn/log/rebootinfo.txt

#
date >> /home/local/Autonomous_Robot_Car/svn/log/rebootinfo.txt
exit 0
