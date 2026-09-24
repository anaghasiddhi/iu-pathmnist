
from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "output" / "pathmnist.duckdb"

# Avoid accidentally creating a new, empty database.
if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

with duckdb.connect(str(DB_PATH), read_only=True) as con:
    print("TOTAL ROWS")
    print(con.execute("""
        SELECT COUNT(*)
        FROM image_metadata
    """).fetchall())

    print("\nRECORDS PER SPLIT")
    print(con.execute("""
        SELECT split, COUNT(*) AS image_count
        FROM image_metadata
        GROUP BY split
        ORDER BY split
    """).df())

    print("\nLOW-VARIANCE FLAGS")
    print(con.execute("""
        SELECT split, COUNT(*) AS flagged_images
        FROM image_metadata
        WHERE low_variance_flag = TRUE
        GROUP BY split
        ORDER BY split
    """).df())

    print("\nFIRST FIVE RECORDS")
    print(con.execute("""
        SELECT
            image_id,
            split,
            label,
            width,
            height,
            mean_intensity,
            std_intensity
        FROM image_metadata
        LIMIT 5
    """).df())
    
    print("\nFINAL DATA QUALITY CHECKS")

    result = con.execute("""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT image_id) AS unique_ids,
            COUNT(*) FILTER (
                WHERE mean_intensity NOT BETWEEN 0 AND 255
                   OR min_intensity NOT BETWEEN 0 AND 255
                   OR max_intensity NOT BETWEEN 0 AND 255
                   OR std_intensity < 0
                   OR NOT isfinite(mean_intensity)
                   OR NOT isfinite(std_intensity)
            ) AS invalid_statistics,
            COUNT(*) FILTER (
                WHERE width != 28 OR height != 28
            ) AS invalid_dimensions
        FROM image_metadata
    """).df()

    print(result.to_string(index=False))