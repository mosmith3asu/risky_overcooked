# Devel
To run local server with Flask (no docker image)
```bash
python ./server/app.py
```


# Docker Scripts

Force recreate
```bash
./up.sh production
```
Delete all build records
```bash
docker buildx history rm --all
docker buildx prune --all
```
data stored in:
```
C:\app\data\*.pkl
```

# AWS
if up.sh not running `sudo chmod u+r+x up.sh`

# Notes
- may have to remove `graphics.js` from `static/js/` when using docker. this was added manually
- Binded local repo in docker-compose.yml