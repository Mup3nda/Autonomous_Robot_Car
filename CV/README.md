## Ball trancking script
2. First stop the raspi which is already running:
```bash
sudo pkill -f stream_server 
```
2. Then activate the virtual env:
```bash
cd ~/Autonomous_Robot_Car
source source venv/bin/activate
```
3. Run the script on Raspberry Pi:
```bash
python3 CV/ball_tracking.py 
```
