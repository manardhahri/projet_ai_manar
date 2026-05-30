CREATE DATABASE IF NOT EXISTS skin_cancer_db;
USE skin_cancer_db;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    result VARCHAR(20) NOT NULL,
    probability FLOAT NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    gradcam_path VARCHAR(255),
    heatmap_path VARCHAR(255),
    attention_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO users (username, password)
VALUES ('admin', '1234')
ON DUPLICATE KEY UPDATE username = username;
