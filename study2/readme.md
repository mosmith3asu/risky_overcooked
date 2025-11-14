

# Import Data from AWS
To import files from webserver:
```bash
scp -r -i ./human_data/risky_overcooked.pem ec2-user@54.234.124.182:/app/data ./human_data/
```
```bash
scp -r -i Downloads/risky_overcooked.pem ec2-user@54.234.124.182:/app/data ./human_data/

```
```bash
runas /user:Administrator "scp -r -i ./human_data/risky_overcooked.pem ec2-user@54.234.124.182:/app/data ./human_data/"
```

If get error "It is required that your private key files are NOT accessible by others."
```bash
chmod 400 ./human_data/risky_overcooked.pem
```