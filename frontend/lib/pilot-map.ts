import type { StyleSpecification } from 'maplibre-gl'

/** Greater Chennai / GCC pilot (File 12/13 bbox: 12.85–13.25 N, 80.10–80.35 E). */
export const PILOT_CENTER: [number, number] = [80.225, 13.05]
export const PILOT_ZOOM = 11
export const SITE_ZOOM = 14

/** OSM-based vector tiles via OpenFreeMap (no API key). Dark style matches the dashboard. */
export const OPEN_TILE_STYLE_URL = 'https://tiles.openfreemap.org/styles/dark'

/** OSM-based raster fallback (Carto Dark Matter, OSM data). */
export const OSM_RASTER_STYLE: StyleSpecification = {
  version: 8,
  name: 'OSM raster',
  sources: {
    osm: {
      type: 'raster',
      tiles: ['https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'],
      tileSize: 256,
      attribution: '© OpenStreetMap contributors © CARTO',
      maxzoom: 19,
    },
  },
  layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
}
