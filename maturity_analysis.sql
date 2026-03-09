-- ============================================================
-- MATURITY ANALYSIS DATABASE
-- Schema: Personality Trait → Maturity Scoring System
-- Dimensions: Emotional, Social, Self-Awareness, Communication
-- Score: 1 (Very Immature) → 10 (Very Mature)
-- ============================================================

-- ─────────────────────────────────────────────
-- TABLE 1: PERSONS
-- ─────────────────────────────────────────────
CREATE TABLE persons (
    person_id       INT PRIMARY KEY,
    label           VARCHAR(50) NOT NULL,
    gender_note     VARCHAR(100),
    summary         VARCHAR(255)
);

INSERT INTO persons VALUES
(1, 'Person 1', 'Female', 'Too feminine, gossip-prone, narcissistic tendencies'),
(2, 'Person 2', 'Female, masculine-presenting', 'Holds hostility toward men, gender-identity conflict'),
(3, 'Person 3', 'Not specified', 'Autistic, shy, socially isolated'),
(4, 'Person 4', 'Not specified', 'Chaotic behaviour, low accountability, poor judgment'),
(5, 'Person 5', 'Not specified', 'Validation-seeking, entitlement complex');


-- ─────────────────────────────────────────────
-- TABLE 2: RAW TRAITS
-- ─────────────────────────────────────────────
CREATE TABLE traits (
    trait_id        INT PRIMARY KEY,
    person_id       INT REFERENCES persons(person_id),
    trait_label     VARCHAR(100),
    trait_category  VARCHAR(50)  -- emotional / social / self_awareness / communication
);

INSERT INTO traits VALUES
-- Person 1
(101, 1, 'Excessive femininity used manipulatively',   'social'),
(102, 1, 'Habitual gossip mongering',                  'communication'),
(103, 1, 'Narcissistic traits (self-centred, vain)',   'self_awareness'),
(104, 1, 'Low empathy for others',                     'emotional'),

-- Person 2
(201, 2, 'Masculine self-presentation, identity tension','self_awareness'),
(202, 2, 'Misandry / blanket hatred of men',           'emotional'),
(203, 2, 'Likely poor cross-gender communication',     'communication'),
(204, 2, 'Possible unresolved emotional pain',         'emotional'),

-- Person 3
(301, 3, 'Autism spectrum traits',                     'social'),
(302, 3, 'Social shyness and withdrawal',              'social'),
(303, 3, 'Perceived as socially atypical (weirdo)',    'communication'),
(304, 3, 'Likely high internal self-awareness',        'self_awareness'),

-- Person 4
(401, 4, 'Chaotic decision-making',                    'emotional'),
(402, 4, 'Chronic lack of accountability',             'self_awareness'),
(403, 4, 'Poor judgment / impulsivity',                'emotional'),
(404, 4, 'Disruptive communication style',             'communication'),

-- Person 5
(501, 5, 'Excessive validation-seeking',               'social'),
(502, 5, 'Sense of entitlement',                       'self_awareness'),
(503, 5, 'Emotionally reactive to criticism',          'emotional'),
(504, 5, 'Performs rather than communicates genuinely','communication');


-- ─────────────────────────────────────────────
-- TABLE 3: MATURITY SCORES (1=Very Immature, 10=Very Mature)
-- ─────────────────────────────────────────────
CREATE TABLE maturity_scores (
    score_id                INT PRIMARY KEY,
    person_id               INT REFERENCES persons(person_id),
    emotional_maturity      INT CHECK (emotional_maturity BETWEEN 1 AND 10),
    social_maturity         INT CHECK (social_maturity BETWEEN 1 AND 10),
    self_awareness          INT CHECK (self_awareness BETWEEN 1 AND 10),
    communication_style     INT CHECK (communication_style BETWEEN 1 AND 10),
    overall_avg             DECIMAL(4,2)
);

INSERT INTO maturity_scores VALUES
-- person_id, emotional, social, self_awareness, communication, avg
(1, 1, 3, 2, 2, 2.00),
(2, 2, 3, 3, 3, 2.75),
(3, 6, 4, 7, 4, 5.25),
(4, 1, 2, 1, 2, 1.50),
(5, 2, 3, 2, 3, 2.50);


-- ─────────────────────────────────────────────
-- TABLE 4: MATURITY CLASSIFICATION
-- ─────────────────────────────────────────────
CREATE TABLE maturity_classification (
    class_id        INT PRIMARY KEY,
    person_id       INT REFERENCES persons(person_id),
    overall_avg     DECIMAL(4,2),
    verdict         VARCHAR(20),  -- 'Immature', 'Borderline', 'Mature'
    confidence      VARCHAR(10),  -- 'High', 'Medium', 'Low'
    key_reason      VARCHAR(255)
);

INSERT INTO maturity_classification VALUES
(1, 1, 2.00, 'Immature',    'High',   'Narcissism and gossip are core markers of emotional and social immaturity'),
(2, 2, 2.75, 'Immature',    'High',   'Blanket hatred of a group signals unresolved trauma, not maturity'),
(3, 3, 5.25, 'Borderline',  'Low',    'Autism affects social output, NOT emotional depth — unfair to classify as immature'),
(4, 4, 1.50, 'Immature',    'High',   'Lowest scorer — chaos + zero accountability = clearest immaturity signal'),
(5, 5, 2.50, 'Immature',    'High',   'Entitlement and validation-seeking indicate arrested emotional development');


-- ─────────────────────────────────────────────
-- ANALYTICAL QUERIES
-- ─────────────────────────────────────────────

-- Q1: Full maturity profile per person
SELECT
    p.label,
    s.emotional_maturity,
    s.social_maturity,
    s.self_awareness,
    s.communication_style,
    s.overall_avg,
    c.verdict,
    c.confidence
FROM persons p
JOIN maturity_scores s ON p.person_id = s.person_id
JOIN maturity_classification c ON p.person_id = c.person_id
ORDER BY s.overall_avg ASC;

-- Q2: Ranked from most to least mature
SELECT
    p.label,
    s.overall_avg,
    c.verdict,
    RANK() OVER (ORDER BY s.overall_avg DESC) AS maturity_rank
FROM persons p
JOIN maturity_scores s ON p.person_id = s.person_id
JOIN maturity_classification c ON p.person_id = c.person_id;

-- Q3: Dimension breakdown — where does each person fail most?
SELECT
    p.label,
    LEAST(s.emotional_maturity, s.social_maturity, s.self_awareness, s.communication_style) AS weakest_score,
    CASE
        WHEN s.emotional_maturity = LEAST(s.emotional_maturity, s.social_maturity, s.self_awareness, s.communication_style)
            THEN 'Emotional Maturity'
        WHEN s.social_maturity = LEAST(s.emotional_maturity, s.social_maturity, s.self_awareness, s.communication_style)
            THEN 'Social Maturity'
        WHEN s.self_awareness = LEAST(s.emotional_maturity, s.social_maturity, s.self_awareness, s.communication_style)
            THEN 'Self-Awareness'
        ELSE 'Communication Style'
    END AS biggest_weakness
FROM persons p
JOIN maturity_scores s ON p.person_id = s.person_id;

-- Q4: Group average across all persons per dimension
SELECT
    ROUND(AVG(emotional_maturity), 2)  AS avg_emotional,
    ROUND(AVG(social_maturity), 2)     AS avg_social,
    ROUND(AVG(self_awareness), 2)      AS avg_self_awareness,
    ROUND(AVG(communication_style), 2) AS avg_communication,
    ROUND(AVG(overall_avg), 2)         AS group_overall_avg
FROM maturity_scores;

-- Q5: Verdict count summary
SELECT verdict, COUNT(*) AS count
FROM maturity_classification
GROUP BY verdict;
