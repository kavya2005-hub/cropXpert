-- ============================================================
-- CropXpert AI — Database Upgrade Script
-- Run this once in MySQL before starting app.py
-- ============================================================

USE cropxpert;

-- ── Phase 3: Recommendation History ──────────────────────────
CREATE TABLE IF NOT EXISTS recommendations (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    user_id     INT           NOT NULL,
    city        VARCHAR(100)  DEFAULT NULL,
    climate     VARCHAR(50)   NOT NULL,
    soil_type   VARCHAR(50)   NOT NULL,
    water_level VARCHAR(20)   NOT NULL,
    land_size   FLOAT         NOT NULL,
    crop_name   VARCHAR(150)  NOT NULL,
    confidence  FLOAT         NOT NULL DEFAULT 0,
    fertilizer  VARCHAR(100)  DEFAULT NULL,
    disease     VARCHAR(100)  DEFAULT NULL,
    medicine    VARCHAR(100)  DEFAULT NULL,
    temperature FLOAT         DEFAULT NULL,   -- from Phase 2 weather
    humidity    FLOAT         DEFAULT NULL,   -- from Phase 2 weather
    created_at  TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at)
);

-- ── Phase 2: Weather Cache (optional optimisation) ───────────
-- Stores fetched weather data to avoid hitting OpenWeather API
-- more than once per city per hour.
CREATE TABLE IF NOT EXISTS weather_cache (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    city         VARCHAR(100) NOT NULL,
    temperature  FLOAT        NOT NULL,
    humidity     FLOAT        NOT NULL,
    description  VARCHAR(100) NOT NULL,
    climate      VARCHAR(50)  NOT NULL,   -- derived climate label
    fetched_at   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_city (city),
    INDEX idx_fetched_at (fetched_at)
);
