# Autonomous_Robot_Car

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

## Git Setup on Robot
Before committing, set your identity:
```bash
git config user.name "Your Name"
git config user.email "your@email.com"
git commit -m "your message"
```

This way, **nobody can commit without explicitly identifying themselves** first!