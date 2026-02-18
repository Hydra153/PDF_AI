# 1. Clone the repo

git clone https://github.com/Hydra153/PDF_AI.git
cd PDF_AI

# 2. Start everything

docker compose up --build

# First run: builds image (~5 min) + downloads models (~10-15 min)

# Every run after: starts in ~30 seconds

---

Future workflow (after making changes on either machine):

# Push changes from this PC:

git add .
git commit -m "description of what you changed"
git push

# Pull changes on the other PC:

git pull
docker compose up --build
