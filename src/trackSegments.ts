export interface TrackPoint { x: number; y: number; transfer?: boolean; gap_before?: boolean; time_s?: number; segment_id?: number }
type LineData = (number[] | null)[]

// Each edge belongs to exactly one series; null prevents ECharts bridging gaps.
export function splitTrack(points: TrackPoint[]) {
  const lines: Record<'track' | 'transfer' | 'gap', LineData> = { track: [], transfer: [], gap: [] }
  let previous: keyof typeof lines | undefined
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i]!, b = points[i + 1]!
    const kind = b.gap_before ? 'gap' : a.transfer ? 'transfer' : 'track'
    const target = lines[kind]
    if (kind !== previous) {
      if (target.length) target.push(null)
      target.push([a.x, a.y])
    }
    target.push([b.x, b.y])
    previous = kind
  }
  if (points.length === 1) lines.track.push([points[0]!.x, points[0]!.y])
  return lines
}

export const gapLineStyle = { color: '#8d9da4', opacity: 0.32, width: 0.7, type: [3, 12] }
