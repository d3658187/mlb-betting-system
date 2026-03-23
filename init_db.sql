-- MLB Betting System - PostgreSQL schema init
-- Run: psql "$DATABASE_URL" -f init_db.sql

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Teams
CREATE TABLE IF NOT EXISTS teams (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  mlb_team_id INTEGER UNIQUE,
  name TEXT NOT NULL,
  abbreviation TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Games / schedule
CREATE TABLE IF NOT EXISTS games (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  mlb_game_id BIGINT UNIQUE,
  game_date DATE NOT NULL,
  game_datetime TIMESTAMPTZ,
  home_team_id UUID REFERENCES teams(id),
  away_team_id UUID REFERENCES teams(id),
  venue TEXT,
  status TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Betting odds (moneyline, spread, total)
CREATE TABLE IF NOT EXISTS odds (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  game_id UUID NOT NULL REFERENCES games(id) ON DELETE CASCADE,
  sportsbook TEXT NOT NULL,
  market TEXT NOT NULL, -- moneyline | spread | total
  selection TEXT NOT NULL, -- home | away | over | under | team name
  price INTEGER NOT NULL, -- American odds
  line NUMERIC(6,2), -- spread/total line
  retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (game_id, sportsbook, market, selection, line, retrieved_at)
);

-- Basic game results (for backtesting)
CREATE TABLE IF NOT EXISTS game_results (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  game_id UUID NOT NULL REFERENCES games(id) ON DELETE CASCADE,
  home_score INTEGER,
  away_score INTEGER,
  home_win BOOLEAN,
  total_points INTEGER,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (game_id)
);

-- Game weather (Open-Meteo)
CREATE TABLE IF NOT EXISTS game_weather (
  id BIGSERIAL PRIMARY KEY,
  mlb_game_id BIGINT NOT NULL,
  game_datetime TIMESTAMPTZ,
  venue TEXT,
  latitude DOUBLE PRECISION,
  longitude DOUBLE PRECISION,
  temperature_c NUMERIC(6,2),
  relative_humidity NUMERIC(6,2),
  wind_speed NUMERIC(6,2),
  wind_direction NUMERIC(6,2),
  data_time TIMESTAMPTZ,
  source TEXT DEFAULT 'open-meteo',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (mlb_game_id)
);

-- ETL run log
CREATE TABLE IF NOT EXISTS etl_runs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  run_date DATE NOT NULL,
  status TEXT NOT NULL, -- success | failed
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ,
  message TEXT
);

-- Optional indexes
CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_odds_game ON odds(game_id);
CREATE INDEX IF NOT EXISTS idx_odds_market ON odds(market);
CREATE INDEX IF NOT EXISTS idx_game_weather_mlb_game_id ON game_weather(mlb_game_id);
