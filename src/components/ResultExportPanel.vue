<template>
  <el-card shadow="never" class="export-panel">
    <template #header><strong>{{ t('Export results') }}</strong></template>
    <p class="export-note">{{ t(kind === 'simulation' ? 'Top 10 uses the saved criterion and currently selected time budget. All includes failed candidates and their reasons. Exporting does not rerun the calculation.' : 'This run contains one uploaded track and per-beacon estimates. Top 10 sorts successful beacon results by residual RMS; it is not a trajectory or coordinate-accuracy ranking.') }}</p>
    <el-radio-group v-model="scope" :aria-label="t('Export scope')">
      <el-radio-button value="all">{{ t('All data') }}</el-radio-button>
      <el-radio-button value="top10">{{ t(kind==='simulation'?'Top 10 routes':'Top 10 beacon results') }}</el-radio-button>
      <el-radio-button value="custom">{{ t('Choose results') }}</el-radio-button>
    </el-radio-group>
    <div v-if="scope==='custom'" class="selection">
      <el-input v-model="query" :placeholder="t('Search results')" clearable />
      <el-table :data="filteredRows" :row-key="rowId" max-height="300" @selection-change="selectRows">
        <el-table-column type="selection" :reserve-selection="true" width="45"/>
        <el-table-column :label="t(kind==='simulation'?'Route':'Beacon')" min-width="160"><template #default="{row}">{{ kind==='simulation' ? `${t(row.label)} · ${Number(row.size).toFixed(0)} m · ${row.angle_deg}°` : `B${row.beacon_id}` }}</template></el-table-column>
        <el-table-column :label="t(kind==='simulation'?criterion.toUpperCase()+' / m²':'Residual RMS / ms')" min-width="110"><template #default="{row}">{{ metric(row) }}</template></el-table-column>
        <el-table-column :label="t('Status / reason')" min-width="180"><template #default="{row}">{{ t(row.validation_status==='failed' ? 'Validation failed' : row.reason ?? row.message ?? (kind==='simulation'?(row.valid?'Valid designs':'Failed'):row.status)) }}<span v-if="row.validation_status==='failed'">: {{ t(row.validation_reason) }}</span></template></el-table-column>
      </el-table>
      <p class="export-note">{{ t('Selected results:') }} {{ selectedIds.length }}</p>
    </div>
    <el-checkbox-group v-model="parts" class="parts" :aria-label="t('Export contents')">
      <el-checkbox value="trajectory">{{ t('Track coordinates') }}</el-checkbox>
      <el-checkbox value="observations">{{ t('Observation data') }}</el-checkbox>
      <el-checkbox value="residuals">{{ t('Residuals') }}</el-checkbox>
      <el-checkbox v-if="kind==='field' || hasRaw" value="raw">{{ t('Original uploaded files') }}</el-checkbox>
    </el-checkbox-group>
    <p class="export-note">{{ t('Summary CSV, settings and JSON are always included. The ZIP contains a folder for each selected result.') }} <span v-if="kind==='field' && parts.includes('raw')">{{ t('Original files remain complete, including rows for unselected beacons.') }}</span></p>
    <el-button type="primary" :loading="downloading" :disabled="!taskId || (scope==='custom'&&!selectedIds.length)" @click="download">{{ t('Download ZIP') }}</el-button>
  </el-card>
</template>
<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { t } from '../i18n'
import { apiClient } from '../api'
const props=defineProps<{taskId:string,kind:'simulation'|'field',rows:any[],budgetH?:number,criterion?:string,hasRaw?:boolean}>()
const scope=ref('all'),parts=ref(['trajectory','observations','residuals']),query=ref(''),selectedIds=ref<string[]>([]),downloading=ref(false)
const criterion=computed(()=>props.criterion??'c3')
const rowId=(row:any)=>String(props.kind==='simulation'?row.candidate_id:row.beacon_id)
const filteredRows=computed(()=>props.rows.filter(row=>`${t(row.label??'')} ${rowId(row)} ${row.size??''} ${row.angle_deg??''}`.toLowerCase().includes(query.value.toLowerCase())))
const metric=(row:any)=>{const n=props.kind==='simulation'?row[criterion.value+'_m2']:row.rmse_time_ms;return typeof n==='number'&&Number.isFinite(n)?n.toFixed(6):'—'}
function selectRows(rows:any[]){selectedIds.value=rows.map(rowId)}
watch(scope,()=>{selectedIds.value=[];query.value=''})
watch(()=>props.taskId,()=>{selectedIds.value=[];scope.value='all';query.value=''})
async function download(){downloading.value=true;try{const response=await apiClient.post(`/api/export/${props.taskId}`,{scope:scope.value,ids:selectedIds.value,parts:parts.value,budget_h:props.budgetH},{responseType:'blob',timeout:180000});const url=URL.createObjectURL(response.data);const a=document.createElement('a');a.href=url;a.download=`seaarray-${props.kind}-${scope.value}.zip`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);ElMessage.success(t('Export download started.'))}catch(error:any){let message='Export failed';if(error.response?.data instanceof Blob){try{message=JSON.parse(await error.response.data.text()).detail??message}catch{}}else message=error.message??message;ElMessage.error(t(message))}finally{downloading.value=false}}
</script>
<style scoped>
.export-panel{margin:20px 0;border:1px solid #dfe7e9;border-radius:10px}.export-note{font-size:12px;line-height:1.7;color:#647982}.selection{margin-top:16px}.selection .el-input{margin-bottom:10px}.parts{margin-top:16px;display:flex;flex-wrap:wrap;gap:0 12px}.export-panel :deep(.el-radio-group){display:flex;flex-direction:row;flex-wrap:wrap;gap:6px}
</style>
