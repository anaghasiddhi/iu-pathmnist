
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "output" / "pathmnist.duckdb"
SQL_PATH = PROJECT_ROOT / "sql" / "analysis.sql"

if not DB_PATH.exists():
    raise FileNotFoundError(DB_PATH)

query = SQL_PATH.read_text(encoding="utf-8")

with duckdb.connect(str(DB_PATH), read_only=True) as con:
    results = con.execute(query).df()

print("\nCLASS BALANCE")
print(results.to_string(index=False))

# Save the analysis results.
output_path = PROJECT_ROOT / "output" / "class_balance.csv"
results.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")


# Part 2B: Compare image characteristics.
characteristics_sql = (
    PROJECT_ROOT / "sql" / "image_characteristics.sql"
).read_text(encoding="utf-8")

with duckdb.connect(str(DB_PATH), read_only=True) as con:
    characteristics = con.execute(characteristics_sql).df()

print("\nIMAGE CHARACTERISTICS")
print(characteristics.to_string(index=False))

characteristics.to_csv(
    PROJECT_ROOT / "output" / "image_characteristics.csv",
    index=False
)


# Part 2C: Identify quantitatively unusual images.
unusual_sql = (
    PROJECT_ROOT / "sql" / "unusual_images.sql"
).read_text(encoding="utf-8")

with duckdb.connect(str(DB_PATH), read_only=True) as con:
    unusual = con.execute(unusual_sql).df()

print("\nUNUSUAL IMAGES")
print(unusual.to_string(index=False))

unusual.to_csv(
    PROJECT_ROOT / "output" / "unusual_images.csv",
    index=False
)