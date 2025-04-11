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