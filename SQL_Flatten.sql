CREATE TABLE flat_records AS
SELECT over_50k,
       r.id AS id,
       age,
       workclass_id,
       w.name AS workclass,
       education_level_id,
       e.name AS education_level,
       education_num,
       marital_status_id,
       m.name AS marital_status,
       occupation_id,
       o.name AS occupation,
       relationship_id,
       re.name AS relationship,
       race_id,
       ra.name AS race,
       sex_id,
       s.name AS sex,
       capital_gain,
       capital_loss,
       hours_week,
       country_id,
       c.name AS country
FROM records AS r
    INNER JOIN workclasses AS w
        ON r.workclass_id = w.id
    INNER JOIN education_levels AS e
        ON r.education_level_id = e.id
    INNER JOIN marital_statuses AS m
        ON r.marital_status_id = m.id
    INNER JOIN occupations AS o
        ON r.occupation_id = o.id
    INNER JOIN relationships AS re
        ON r.relationship_id = re.id
    INNER JOIN races AS ra
        ON r.race_id = ra.id
    INNER JOIN sexes AS s
        ON r.sex_id = s.id
    INNER JOIN countries AS c
        ON r.country_id = c.id;
