#!/bin/bash
# echo -e "\nStarting mission\n"
# ----------START THE TEST----------------
# cd /home/local/svn/robobot/mqtt_python
# /usr/bin/python3 mqtt-client.py -n >>log_out.txt 2>>log_err.txt &

# -----------START OUR FINAL MISSION--------------
/usr/bin/python3 -u /home/local/svn/robobot/mqtt_python/Missions/final_competition_mission.py >>log_out.txt 2>>log_err.txt &

# echo "mission ended"
exit 0
