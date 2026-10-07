<!--
  RealResidualChart.vue - Residual series散点图（可配置标题和单位）
  X 轴：历元序号, Y 轴：Residual
  用于展示时延Residual (ms) 或等效距离Residual (m)
  响应式：窗口 resize 自动重绘
-->
<template>
  <div class="real-residual-chart">
    <div ref="chartRef" class="chart-box"></div>
  </div>
</template>

<script setup lang="ts">
import { t, locale, setLocalizedOption } from "../i18n"
import { onMounted, onUnmounted, ref, watch } from 'vue'
import * as echarts from 'echarts'

interface ResidualPoint {
  epoch: number    // 历元序号
  value: number    // Residual
}

const props = defineProps<{
  data: ResidualPoint[]
  outlierMask?: boolean[]   // 异常点标记 (true=异常)
  title?: string            // 图表标题，默认 "Residual series"
  yAxisName?: string        // Y 轴单位，默认 "Residual"
  unit?: string             // tooltip 显示单位，默认 ""
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

watch(() => [props.data, props.outlierMask], () => updateChart(), { deep: true })

function updateChart() {
  if (!chart || chart.isDisposed() || !props.data.length) return

  const normalData: [number, number][] = []
  const outlierData: [number, number][] = []

  let sum = 0; let maxAbs = 0
  props.data.forEach((p, i) => {
    const isOutlier = props.outlierMask?.[i] === true
    if (isOutlier) {
      outlierData.push([p.epoch, p.value])
    } else {
      normalData.push([p.epoch, p.value])
    }
    sum += p.value
    if (Math.abs(p.value) > maxAbs) maxAbs = Math.abs(p.value)
  })
  const mean = sum / props.data.length

  // 自动计算 Y 轴范围 (±3σ 或 ±maxAbs)
  const values = props.data.map(p => p.value)
  const std = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length)
  const yMax = Math.max(maxAbs * 1.15, std * 4, 1)

  const title = props.title ?? 'Residual series'
  const yAxisName = props.yAxisName ?? 'Residual'
  const unit = props.unit ?? ''

  setLocalizedOption(chart,{
    title: { text: title, left: 'center', top: 4, textStyle: { fontSize: 14, fontWeight: 600 } },
    tooltip: { trigger: 'axis', formatter: (params: any) => {
      const p = Array.isArray(params) ? params[0] : params
      if (!p?.data) return ''
      return `Epoch #${p.data[0]}<br/>Residual: ${p.data[1].toFixed(4)} ${unit}`
    }},
    legend: { data: ['Accepted', 'Rejected outliers'], bottom: 0, textStyle: { fontSize: 11 } },
    grid: { left: 55, right: 30, top: 55, bottom: 45 },
    xAxis: { type: 'value', name: 'Epoch', nameLocation: 'center', nameGap: 25 },
    yAxis: { type: 'value', name: yAxisName, nameLocation: 'center', nameGap: 38,
      min: -yMax, max: yMax,
      splitLine: { lineStyle: { type: 'dashed', color: '#e4e7ed' } }
    },
    series: [
      {
        name: 'Accepted', type: 'scatter', data: normalData,
        symbolSize: 4, itemStyle: { color: '#409eff', borderColor: '#409eff', borderWidth: 1 },
      },
      {
        name: 'Rejected outliers', type: 'scatter', data: outlierData,
        symbolSize: 6, itemStyle: { color: '#f56c6c', borderColor: '#c45656', borderWidth: 2 },
      },
      {
        name: 'Zero', type: 'line',
        data: [[0, 0], [props.data.length, 0]],
        symbol: 'none', lineStyle: { color: '#999', type: 'dashed', width: 1 },
        silent: true,
      },
    ],
  }, { notMerge: true })
}
watch(locale,()=>updateChart())
</script>

<style scoped>
.real-residual-chart { width: 100%; }
.chart-box { width: 100%; height: 350px; }
</style>