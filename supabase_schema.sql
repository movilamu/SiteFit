-- SiteFit — Supabase schema
-- Run this once in the Supabase SQL editor (Project → SQL Editor → New query).

-- 1. Candidate sites -----------------------------------------------------
-- Read by api/sites_routes.py via the REST API (GET /sites).
create table if not exists candidate_sites (
    id                bigint generated always as identity primary key,
    site_id           text unique,               -- falls back to `id` if null
    name              text,
    latitude          double precision not null,
    longitude         double precision not null,
    unit_size         double precision,           -- sq ft
    rent              double precision,           -- monthly rent
    tenure_type       text,                        -- e.g. 'lease', 'freehold'
    status            text default 'active',       -- only 'active' rows are served by default
    attractiveness    double precision,
    metadata          jsonb default '{}'::jsonb    -- any extra scoring criteria columns
);

create index if not exists idx_candidate_sites_status on candidate_sites (status);

-- 2. Saved MCA scenarios --------------------------------------------------
-- Read/written by api/scenario_routes.py (POST/GET /scenarios) and
-- api/export_routes.py.
create table if not exists saved_scenarios (
    scenario_id       bigint generated always as identity primary key,
    name              text not null,
    weight_set        jsonb not null,             -- {"criterion": weight, ...} sums to 1.0
    aggregation_method text not null,              -- 'weighted_sum' | 'weighted_product' | 'distance_to_ideal'
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now()
);

-- 3. Export records --------------------------------------------------------
-- Written by api/export_routes.py after a shortlist file is uploaded to
-- Storage (see bucket below).
create table if not exists export_records (
    export_id         bigint generated always as identity primary key,
    scenario_id       bigint references saved_scenarios (scenario_id),
    storage_path      text,
    created_at        timestamptz not null default now()
);

-- 4. Storage bucket for generated exports ----------------------------------
-- Name must match SUPABASE_EXPORT_BUCKET (defaults to 'exports').
insert into storage.buckets (id, name, public)
values ('exports', 'exports', false)
on conflict (id) do nothing;

-- 5. Row Level Security -----------------------------------------------------
-- Simplest workable setup for a first deploy: RLS off, access controlled by
-- which key (anon vs service_role) your backend uses. Tighten later with
-- policies once you have real users.
alter table candidate_sites disable row level security;
alter table saved_scenarios disable row level security;
alter table export_records disable row level security;

-- 6. Seed a handful of sample sites so /sites returns something immediately.
insert into candidate_sites
    (site_id, name, latitude, longitude, unit_size, rent, tenure_type, status, attractiveness, metadata)
values
    ('SITE-001', 'Anna Nagar Corner Unit',   13.0850, 80.2101, 1800, 145000, 'lease',     'active', 0.82, '{"footfall_index": 0.71}'),
    ('SITE-002', 'T. Nagar High Street',     13.0410, 80.2337, 2400, 210000, 'lease',     'active', 0.91, '{"footfall_index": 0.88}'),
    ('SITE-003', 'Velachery Mall Kiosk',     12.9791, 80.2183, 600,  60000,  'freehold',  'active', 0.64, '{"footfall_index": 0.55}'),
    ('SITE-004', 'OMR Tech Corridor Unit',   12.8996, 80.2274, 1500, 130000, 'lease',     'active', 0.77, '{"footfall_index": 0.60}'),
    ('SITE-005', 'Adyar Riverside Retail',   13.0067, 80.2570, 1200, 118000, 'lease',     'active', 0.73, '{"footfall_index": 0.66}')
on conflict (site_id) do nothing;
