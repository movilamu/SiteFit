'use client'

import { useEffect, useRef } from 'react'
import maplibregl, { type GeoJSONSource, type Map as MapLibreMap, type Marker } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import './sites-map.css'

import {
  OPEN_TILE_STYLE_URL,
  OSM_RASTER_STYLE,
  PILOT_CENTER,
  PILOT_ZOOM,
  SITE_ZOOM,
} from '@/lib/pilot-map'
import { displayName, siteId, type CandidateSite } from '@/lib/sites'

type SitesMapProps = {
  sites: CandidateSite[]
  selectedSiteId: string | null
  onSelectSite: (id: string) => void
  catchment: GeoJSON.Geometry | null
}

const EMPTY_COLLECTION: GeoJSON.FeatureCollection = {
  type: 'FeatureCollection',
  features: [],
}

function catchmentCollection(geometry: GeoJSON.Geometry | null): GeoJSON.FeatureCollection {
  if (!geometry) return EMPTY_COLLECTION
  return {
    type: 'FeatureCollection',
    features: [{ type: 'Feature', properties: {}, geometry }],
  }
}

function ensureCatchmentLayers(map: MapLibreMap) {
  if (!map.getSource('catchment')) {
    map.addSource('catchment', {
      type: 'geojson',
      data: EMPTY_COLLECTION,
    })
  }
  if (!map.getLayer('catchment-fill')) {
    map.addLayer({
      id: 'catchment-fill',
      type: 'fill',
      source: 'catchment',
      paint: {
        'fill-color': '#22d3ee',
        'fill-opacity': 0.16,
      },
    })
  }
  if (!map.getLayer('catchment-outline')) {
    map.addLayer({
      id: 'catchment-outline',
      type: 'line',
      source: 'catchment',
      paint: {
        'line-color': '#67e8f9',
        'line-width': 2,
        'line-opacity': 0.85,
      },
    })
  }
}

export default function SitesMap({
  sites,
  selectedSiteId,
  onSelectSite,
  catchment,
}: SitesMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<MapLibreMap | null>(null)
  const markersRef = useRef<Map<string, Marker>>(new Map())
  const onSelectRef = useRef(onSelectSite)
  const selectedRef = useRef(selectedSiteId)
  const readyRef = useRef(false)

  onSelectRef.current = onSelectSite
  selectedRef.current = selectedSiteId

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: OPEN_TILE_STYLE_URL,
      center: PILOT_CENTER,
      zoom: PILOT_ZOOM,
      attributionControl: true,
    })
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), 'top-right')
    mapRef.current = map

    const onLoad = () => {
      ensureCatchmentLayers(map)
      readyRef.current = true
      map.resize()
    }

    map.on('load', onLoad)
    map.on('error', (event: { error?: { message?: string } }) => {
      const message = event.error?.message || ''
      if (message.includes('style') || message.includes('fetch')) {
        map.setStyle(OSM_RASTER_STYLE)
      }
    })

    const resize = () => map.resize()
    window.addEventListener('resize', resize)

    return () => {
      window.removeEventListener('resize', resize)
      map.off('load', onLoad)
      markersRef.current.forEach((marker: maplibregl.Marker) => marker.remove())
      markersRef.current.clear()
      map.remove()
      mapRef.current = null
      readyRef.current = false
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    const existing = markersRef.current
    const nextIds = new Set(sites.map((site) => siteId(site)))

    existing.forEach((marker: maplibregl.Marker, id: string) => {
      if (!nextIds.has(id)) {
        marker.remove()
        existing.delete(id)
      }
    })

    for (const site of sites) {
      const id = siteId(site)
      const lngLat: [number, number] = [site.coordinates.longitude, site.coordinates.latitude]
      let marker = existing.get(id)
      if (!marker) {
        const el = document.createElement('button')
        el.type = 'button'
        el.className = 'site-marker'
        el.setAttribute('aria-label', displayName(site))
        el.addEventListener('click', (event) => {
          event.stopPropagation()
          onSelectRef.current(id)
        })
        marker = new maplibregl.Marker({ element: el, anchor: 'bottom' })
          .setLngLat(lngLat)
          .addTo(map)
        existing.set(id, marker)
      } else {
        marker.setLngLat(lngLat)
      }
      marker.getElement().classList.toggle('is-selected', selectedRef.current === id)
    }
  }, [sites])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    markersRef.current.forEach((marker: maplibregl.Marker, id: string) => {
      marker.getElement().classList.toggle('is-selected', id === selectedSiteId)
    })

    const selected = sites.find((site) => siteId(site) === selectedSiteId)
    if (!selected) return

    const fly = () => {
      map.flyTo({
        center: [selected.coordinates.longitude, selected.coordinates.latitude],
        zoom: Math.max(map.getZoom(), SITE_ZOOM),
        essential: true,
        duration: 900,
      })
    }

    if (readyRef.current) fly()
    else map.once('load', fly)
  }, [selectedSiteId, sites])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    const apply = () => {
      ensureCatchmentLayers(map)
      const source = map.getSource('catchment') as GeoJSONSource | undefined
      source?.setData(catchmentCollection(catchment))
    }

    if (readyRef.current && map.isStyleLoaded()) {
      apply()
    } else {
      map.once('load', apply)
    }
  }, [catchment])

  return <div ref={containerRef} className="sites-map absolute inset-0" />
}
