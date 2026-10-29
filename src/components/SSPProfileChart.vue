<!-- Plot original SSP knots: sound speed horizontally, positive depth downward. No smoothing or decimation. -->
<template>
  <div class="ssp-profile-chart">
    <div ref="chartRef" class="chart-box"></div>
  </div>
</template>

<script setup lang="ts">
import { t, locale, setLocalizedOption } from "../i18n"
import { onMounted, onUnmounted, ref, watch } from 'vue'
import * as echarts from 'echarts'

interface SSPPoint {
  depth: number
  sv: number
}

const props = defineProps<{
  rawPoints: SSPPoint[]
  transducerDepth?: number  // Positive transducer depth in metres; displayed as a red line.
}>()

const chartRef = ref<HTMLDivElement>()
let chart: echarts.ECharts | null = null

onMounted(() => {
  if (chartRef.value) {
    chart = echarts.init(chartRef.value)
    window.addEventListener('resize', () => chart?.resize())
    updateChart()
  }
})
onUnmounted(() => { chart?.dispose(); chart = null })

watch(
  () => [props.rawPoints],
  () => updateChart(),
  { deep: true }
)

function updateChart() {
  if (!chart || chart.isDisposed()) return

  const raw = props.rawPoints.map(p => [p.sv, p.depth] as [number, number])

  // Compute X axis range: min-10 → max
  const allSv = raw.map(p => p[0])
  const svMin = allSv.length > 0 ? Math.min(...allSv) : 1480
  const svMax = allSv.length > 0 ? Math.max(...allSv) : 1550
  const xMin = Math.floor(svMin) - 5
  const xMax = Math.ceil(svMax) + 5

  if (raw.length === 0) {
    setLocalizedOption(chart,{
      title: { text: 'Sound-speed profile', left: 'center', textStyle: { fontSize: 14, fontWeight: 600 } },
      graphic: {
        type: 'text', left: 'center', top: 'center',
        style: { text: 'Upload an SSP file to view the profile', fontSize: 14, fill: '#909399' }
      }
    })
    return
  }

  const series: any[] = [{ name: 'Uploaded SSP', type: 'line', data: raw,
    lineStyle: { color: '#003366', width: 2 }, symbol: 'none' }]

  // Transducer depth line
  if (props.transducerDepth != null && props.transducerDepth > 0) {
    const svRange = raw.length > 0 ? [raw[0][0], raw[raw.length - 1][0]] : [1450, 1550]
    series.push({
      name: `Transducer depth (${props.transducerDepth} m)`,
      type: 'line',
      markLine: {
        silent: true,
        symbol: 'none',
        lineStyle: { color: '#e74c3c', type: 'dashed', width: 2 },
        data: [{ yAxis: props.transducerDepth, label: { formatter: `${props.transducerDepth}m` } }],
      },
      data: [],
      z: 5,
    })
    // Also add as horizontal line series
    series.push({
      name: 'Transducer depth',
      type: 'line',
      data: [[Math.min(...raw.map(p => p[0])), props.transducerDepth],
             [Math.max(...raw.map(p => p[0])), props.transducerDepth]],
      lineStyle: { color: '#e74c3c', type: 'dashed', width: 2 },
      symbol: 'none',
      silent: true,
      z: 5,
    })
  }

  setLocalizedOption(chart,{
    title: { text: 'Sound-speed profile', left: 'center', top: 4, textStyle: { fontSize: 14, fontWeight: 600 } },
    tooltip: { trigger: 'axis', formatter: (params: any) => {
      const items = Array.isArray(params) ? params : [params]
      const lines = items.filter((p: any) => p?.data && Array.isArray(p.data) && p.data.length === 2)
        .map((p: any) => `${p.seriesName}: SV ${p.data[0].toFixed(2)} m/s, Depth ${p.data[1].toFixed(2)} m`)
      return lines.join('<br/>') || ''
    }},
    legend: { data: ['Uploaded SSP', 'Transducer depth'].filter(Boolean), bottom: 0, textStyle: { fontSize: 11 } },
    grid: { left: 65, right: 30, top: 55, bottom: 50 },
    xAxis: { type: 'value', name: 'Sound speed (m/s)', nameLocation: 'center', nameGap: 25,
      min: xMin, max: xMax },
    yAxis: { type: 'value', name: 'Depth (m)', nameLocation: 'center', nameGap: 38, inverse: true,
      splitLine: { lineStyle: { type: 'dashed', color: '#e4e7ed' } }
    },
    series,
  }, { notMerge: true })
}
watch(locale,()=>updateChart())
</script>

<style scoped>
.ssp-profile-chart { width: 100%; }
.chart-box { width: 100%; height: 450px; }
</style>
