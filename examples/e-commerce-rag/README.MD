Start the Databases (Redis & ChromaDB)
We use Docker Compose to run Redis Stack (which includes RediSearch) and ChromaDB locally.

```Bash
docker compose up -d
```

The -d flag runs the containers in detached mode so you can continue using your terminal.

Running the Application
Once the databases are running and your virtual environment is activated, execute the main script:

```Bash
python main.py
```

Teardown
When you are finished testing, you can stop and remove the database containers by running:

```Bash
docker compose down
```
