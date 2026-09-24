
-- Select the five darkest, five brightest, and
-- five lowest-contrast images using metadata.

WITH ranked AS (
    SELECT
        image_id,
        split,
        label,
        mean_intensity,
        std_intensity,
        ROW_NUMBER() OVER (
            ORDER BY mean_intensity ASC, image_id
        ) AS dark_rank,
        ROW_NUMBER() OVER (
            ORDER BY mean_intensity DESC, image_id
        ) AS bright_rank,
        ROW_NUMBER() OVER (
            ORDER BY std_intensity ASC, image_id
        ) AS low_contrast_rank
    FROM image_metadata
)
SELECT
    image_id,
    split,
    label,
    mean_intensity,
    std_intensity,
    CASE
        WHEN dark_rank <= 5 THEN 'dark'
        WHEN bright_rank <= 5 THEN 'bright'
        ELSE 'low_contrast'
    END AS selection_reason
FROM ranked
WHERE dark_rank <= 5
   OR bright_rank <= 5
   OR low_contrast_rank <= 5
ORDER BY selection_reason, image_id;