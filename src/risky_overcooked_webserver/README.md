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
Installing Docker
```bash
sudo yum install -y docker
sudo service docker start #start service
sudo usermod -a -G docker ec2-user #allow ec2-user to run docker commands without sudo
# close and open ssh terminal to enable permission

# install docker compose
sudo curl -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose version
```
```bash
sudo yum install -y git # if not installed
sudo git clone https://github.com/mosmith3asu/risky_overcooked.git /risky_overcooked

```
if up.sh not running `sudo chmod u+r+x up.sh`

Copying files:
```bash
scp -r -i risky_overcooked.pem ec2-user@54.234.124.182:/app/data D:/app/

```
# Notes
- may have to remove `graphics.js` from `static/js/` when using docker. this was added manually
- Binded local repo in docker-compose.yml