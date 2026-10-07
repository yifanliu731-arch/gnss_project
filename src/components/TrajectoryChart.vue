<template><div ref="host" class="trajectory-chart" role="img" :aria-label="t(&quot;Complete survey route and seafloor beacons in East and North metres&quot;)"></div></template>
<script setup lang="ts">
import { t, locale, setLocalizedOption } from "../i18n"
import {onMounted,onBeforeUnmount,ref,watch} from 'vue'
import * as echarts from 'echarts'
const props=defineProps<{trajectoryData:Array<{x:number,y:number,transfer?:boolean}>,beacons?:number[][]}>()
const host=ref<HTMLDivElement>();let chart:echarts.ECharts|undefined;let observer:ResizeObserver|undefined
function draw(){if(!chart)return;const track=props.trajectoryData.map(p=>[p.x,p.y]);const beacons=props.beacons??[];const all=[...track,...beacons];if(!all.length){chart.clear();return}
 const solid:any[]=[];const transfers:any[]=[];let previous:boolean|undefined;for(let i=0;i<track.length-1;i++){const isTransfer=Boolean(props.trajectoryData[i].transfer);const target=isTransfer?transfers:solid;if(isTransfer!==previous){if(target.length)target.push(null);target.push(track[i])}target.push(track[i+1]);previous=isTransfer}
 const east=all.map(p=>p[0]),north=all.map(p=>p[1]);const cx=(Math.min(...east)+Math.max(...east))/2,cy=(Math.min(...north)+Math.max(...north))/2
 const half=Math.max(Math.max(...east)-Math.min(...east),Math.max(...north)-Math.min(...north),100)*.6
 const plot=Math.max(120,Math.min(chart.getWidth()-110,chart.getHeight()-95));const left=(chart.getWidth()-plot)/2
 setLocalizedOption(chart,{animation:false,color:['#2b8588','#294f62'],tooltip:{trigger:'item',formatter:(p:any)=>`${p.seriesName}<br/>E: ${Number(p.value[0]).toFixed(1)} m<br/>N: ${Number(p.value[1]).toFixed(1)} m`},legend:{bottom:0,textStyle:{color:'#67808a',fontSize:11}},grid:{left,top:25,width:plot,height:plot},xAxis:{type:'value',min:cx-half,max:cx+half,name:'East (m)',nameLocation:'middle',nameGap:28,axisLabel:{fontSize:10,showMinLabel:false,showMaxLabel:false,formatter:(v:number)=>Math.round(v).toLocaleString("en-US")},splitLine:{lineStyle:{color:'#edf1f3'}}},yAxis:{type:'value',min:cy-half,max:cy+half,name:'North (m)',nameLocation:'middle',nameGap:48,axisLabel:{fontSize:10,showMinLabel:false,showMaxLabel:false,formatter:(v:number)=>Math.round(v).toLocaleString("en-US")},splitLine:{lineStyle:{color:'#edf1f3'}}},series:[{name:'Complete route',type:'line',data:solid,smooth:false,showSymbol:false,lineStyle:{width:1.7}},...(transfers.length?[{name:'Transfer segment',type:'line',data:transfers,smooth:false,showSymbol:false,lineStyle:{width:1.5,type:'dashed',color:'#798d96'}}]:[]),{name:'Seafloor beacons',type:'scatter',data:beacons.map((p,i)=>({value:p.slice(0,2),name:`B${i+1}`})),symbol:'diamond',symbolSize:11,label:{show:true,formatter:'{b}',position:'top',fontSize:11,color:'#294f62'}},...(track.length?[{name:'Start',type:'scatter',data:[track[0]],symbolSize:9,itemStyle:{color:'#c3874d'}},...(Math.hypot(track[0][0]-track[track.length-1][0],track[0][1]-track[track.length-1][1])>1e-4?[{name:'Finish',type:'scatter',data:[track[track.length-1]],symbolSize:9,symbol:'rect',itemStyle:{color:'#556f85'}}]:[])]:[])]},{notMerge:true})}
onMounted(()=>{if(host.value){chart=echarts.init(host.value);observer=new ResizeObserver(()=>{chart?.resize();draw()});observer.observe(host.value);draw()}})
watch(()=>[props.trajectoryData,props.beacons],draw,{deep:true})
onBeforeUnmount(()=>{observer?.disconnect();chart?.dispose()})
watch(locale,()=>draw())
</script>
<style scoped>.trajectory-chart{width:100%;height:430px}@media(max-width:600px){.trajectory-chart{height:350px}}</style>
