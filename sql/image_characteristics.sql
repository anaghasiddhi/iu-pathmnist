
-- Part 2B: Image characteristics by class and split

SELECT
    split,
    label,
    COUNT(*) AS image_count,
    ROUND(AVG(mean_intensity), 3) AS avg_mean_intensity,
    ROUND(AVG(std_intensity), 3) AS avg_pixel_std,
    ROUND(MIN(mean_intensity), 3) AS min_mean_intensity,
    ROUND(MAX(mean_intensity), 3) AS max_mean_intensity
FROM image_metadata
GROUP BY split, label
ORDER BY split, label;
  