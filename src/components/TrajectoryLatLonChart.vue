<!--
  TrajectoryLatLonChart.vue - 经Latitude轨迹图表
  接收轨迹经Latitude数据，使用 ECharts 渲染
  在Reference origin位置标记星形符号
-->
<template>
  <div class="latlon-chart">
    <div ref="chartRef" class="chart-box-latlon" role="img" :aria-label="t('Survey track in longitude and latitude')"></div>
  </div>
</template>

<script setup lang="ts">
import { t, locale, setLocalizedOption } from "../i18n"
import { onMounted, onUnmounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import { splitTrack, gapLineStyle } from '../trackSegments'

interface LatLonPoint {
  lat: number
  lon: number
  gap_before?: boolean
}

const props = withDefaults(defineProps<{
  trajectoryData?: LatLonPoint[]
  refLat?: number
  refLon?: number
  displayMode?: 'smooth' | 'scatter'
}>(), {
  trajectoryData: () => [],
  refLat: 0,
  refLon: 0,
  displayMode: 'smooth',
})

const chartRef = ref<HTMLDivElement>()
let chart: echarts.ECharts | null = null

onMounted(() => {
  if (chartRef.value) {
    chart = echarts.init(chartRef.value)
    updateChart()
  }
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  chart?.dispose()
  chart = null
  window.removeEventListener('resize', handleResize)
})

watch(
  () => [props.trajectoryData, props.refLat, props.refLon, props.displayMode],
  () => updateChart(),
  { deep: true }
)

function handleResize() {
  if (chart && !chart.isDisposed()) chart.resize()
}

function updateChart() {
  if (!chart || chart.isDisposed()) return

  // 空值/类型防护：数据未加载或格式错误时退化为空数组，避免 data.map 崩溃
  const source = Array.isArray(props.trajectoryData) ? props.trajectoryData : []
  const points = source
    .filter((p): p is LatLonPoint => p != null && typeof p.lat === 'number' && typeof p.lon === 'number')
  const trackData = points.map(p => [p.lon, p.lat])
  const { track, gap } = splitTrack(points.map(p=>({...p,x:p.lon,y:p.lat})))
  if (!trackData.length) { chart.clear(); return }

  // Compute true geometric centroid from data (not the rounded reference point)
  let lonMin = Infinity, lonMax = -Infinity
  let latMin = Infinity, latMax = -Infinity
  let sumLon = 0, sumLat = 0
  for (const p of trackData) {
    if (p[0] < lonMin) lonMin = p[0]
    if (p[0] > lonMax) lonMax = p[0]
    if (p[1] < latMin) latMin = p[1]
    if (p[1] > latMax) latMax = p[1]
    sumLon += p[0]
    sumLat += p[1]
  }
  const n = trackData.length || 1
  const trueCenLon = sumLon / n   // true centroid (unrounded)
  const trueCenLat = sumLat / n

  // Symmetric half-range centered on TRUE centroid
  const halfLon = Math.max(Math.abs(lonMax - trueCenLon), Math.abs(trueCenLon - lonMin))
  const halfLat = Math.max(Math.abs(latMax - trueCenLat), Math.abs(trueCenLat - latMin))
  const half = Math.max(halfLon, halfLat) * 1.15 + 0.0005

  setLocalizedOption(chart,{
    animation: false,
    legend: {bottom: 0, textStyle: {fontSize: 11, color: '#67808a'}},
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        // 空值/类型防护：markPoint 的 data 是 {name, coord, value} 对象，
        // line 的 data 是 [lon, lat] 数组；只有数组才可解构。
        const d = params?.data
        if (!Array.isArray(d)) return ''
        const lon = d[0]
        const lat = d[1]
        if (typeof lon !== 'number' || typeof lat !== 'number') return ''
        return `Longitude: ${lon.toFixed(6)}°<br/>Latitude: ${lat.toFixed(6)}°`
      }
    },
    grid: {
      left: 65,
      right: 25,
      top: 35,
      bottom: 65,
      containLabel: true
    },
    xAxis: {
      name: 'Longitude (°)',
      nameLocation: 'center',
      nameGap: 30,
      type: 'value',
      min: trueCenLon - half,
      max: trueCenLon + half,
      axisLine: { lineStyle: { color: '#409eff' } },
      axisLabel: {
        formatter: (v: number) => v.toFixed(4)
      }
    },
    yAxis: {
      name: 'Latitude (°)',
      nameLocation: 'center',
      nameGap: 45,
      type: 'value',
      min: trueCenLat - half,
      max: trueCenLat + half,
      axisLine: { lineStyle: { color: '#409eff' } },
      axisLabel: {
        formatter: (v: number) => v.toFixed(4)
      }
    },
    series: [
      {
        name: props.displayMode==='scatter'?'Survey points':'Survey track',
        type: props.displayMode==='scatter'?'scatter':'line',
        data: props.displayMode==='scatter'?trackData:track,
        smooth: .2,
        connectNulls: false,
        showSymbol: false,
        symbol: 'circle',
        symbolSize: 2.5,
        lineStyle: { color: '#409eff', width: 1.5 },
        itemStyle: { color: '#409eff', opacity: .65 },
        markPoint: {
          symbol: 'pin',
          symbolSize: 36,
          data: [
            {
              name: 'Reference origin',
              coord: [props.refLon, props.refLat],
              value: `(${props.refLat.toFixed(4)}, ${props.refLon.toFixed(4)})`,
              itemStyle: { color: '#e6a23c' }
            }
          ],
          label: {
            show: true,
            formatter: (p: any) => p.value,
            fontSize: 11,
            color: '#303133',
            position: 'top',
            distance: 8
          }
        }
      },
      ...(props.displayMode==='smooth' && gap.length ? [{name:'Gap between segments',type:'line',data:gap,smooth:false,connectNulls:false,showSymbol:false,lineStyle:gapLineStyle,silent:true}] : [])
    ]
  }, { notMerge: true })
}
watch(locale,()=>updateChart())
</script>

<style scoped>
.latlon-chart {
  width: 100%;
}
.chart-box-latlon {
  width: 100%;
  height: 420px;
}
</style>
