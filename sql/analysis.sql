
-- Part 2A: Class balance by dataset split

SELECT
    split,
    label,
    COUNT(*) AS class_count,
    SUM(COUNT(*)) OVER (
        PARTITION BY split
    ) AS total_images,
    ROUND(
        100.0 * COUNT(*) /
        SUM(COUNT(*)) OVER (PARTITION BY split),
        2
    ) AS class_percentage
FROM image_metadata
GROUP BY split, label
ORDER BY split, label;