<template>
  <el-card shadow="never" class="profile-validation">
    <template #header><strong>{{ t('Held-out profile comparison · C1 / C2 / C3') }}</strong></template>
    <p class="note">{{ t('Routes were selected using training profiles only. The same held-out daily profile is then used for all selected routes, with the same estimator and unchanged complete-route durations.') }}</p>
    <div class="stats"><span>{{ t('Perturbation multiplier') }} <b>{{ data.amplitude }}×</b></span><span>{{ t('Training profiles') }} <b>{{ data.training_count }}</b></span><span>{{ t('Dates tested') }} <b>{{ data.dates_evaluated.length }}</b></span><span>{{ t('Failed fits') }} <b>{{ data.failed_fits }} / {{ data.fits }}</b></span></div>
    <p class="note">{{ t('Detailed solution date') }}: {{ data.detail_date }} · {{ t('Risk rank') }} {{ data.risk_rank }}</p>
    <el-alert v-if="data.dates_evaluated.length===1" type="info" :closable="false" :title="t('This is a single-date software test, not the full-period experiment or a statistical proof of superiority.')"/>
    <el-table :data="current" stripe size="small">
      <el-table-column :label="t('Selection')" prop="criterion" width="75"/>
      <el-table-column :label="t('Route')" min-width="125"><template #default="{row}">{{ t(row.label) }}<br/><small>{{ fmt(row.size_m,0) }} m · {{ fmt(row.duration_h,2) }} h</small></template></el-table-column>
      <el-table-column :label="t('Coordinate MSE / m²')" min-width="140"><template #default="{row}">{{ fmt(row.mean_mse_m2,5) }}<br/><small v-if="row.mean_scope==='successful_dates_only'">{{ t('Successful dates only') }}</small></template></el-table-column>
      <el-table-column :label="t('Successful / attempted')" min-width="125"><template #default="{row}">{{ row.successful }} / {{ row.attempted }}</template></el-table-column>
      <el-table-column :label="t('Missing truth rays')" prop="missing_rays" min-width="120"/>
      <el-table-column width="100"><template #default="{row}"><el-button size="small" :disabled="!row.id" @click="$emit('inspect',row.id)">{{ t('Inspect route') }}</el-button></template></el-table-column>
    </el-table>
    <p v-if="comparison?.paired_dates" class="note">{{ t('Paired successful dates') }}: {{ comparison.paired_dates }} · C1 {{ fmt(comparison.paired_c1_mse_m2,5) }} / C3 {{ fmt(comparison.paired_c3_mse_m2,5) }} m².</p>
    <p class="note">{{ t('MSE averages squared errors over 3N coordinates. A lower C3 design score is not a guarantee of lower test error. Missing arrivals are counted; a failed selected route is never replaced after testing.') }}</p>
    <el-collapse><el-collapse-item :title="t('Validation trials and failure reasons')"><el-table :data="trials" max-height="350" size="small"><el-table-column prop="date" :label="t('Date')" width="105"/><el-table-column prop="criterion" :label="t('Selection')" min-width="100"/><el-table-column :label="t('Coordinate MSE / m²')" width="140"><template #default="{row}">{{ fmt(row.coordinate_mse_m2,5) }}</template></el-table-column><el-table-column :label="t('Retained / planned')" width="130"><template #default="{row}">{{ row.retained_observations ?? '—' }} / {{ row.planned_observations ?? '—' }}</template></el-table-column><el-table-column :label="t('Status / reason')" min-width="180"><template #default="{row}">{{ t(row.success?'Success':row.reason) }}</template></el-table-column></el-table></el-collapse-item></el-collapse>
  </el-card>
</template>
<script setup lang="ts">
import {computed} from 'vue'
import {t} from '../i18n'
const props=defineProps<{data:any,budgetIndex:number}>()
defineEmits<{inspect:[id:string]}>()
const comparison=computed(()=>props.data.comparisons[props.budgetIndex])
const current=computed(()=>['c1','c2','c3'].map(key=>({...comparison.value?.[key],criterion:key.toUpperCase(),id:comparison.value?.[key+'_id']})))
const trials=computed(()=>props.data.trials.filter((r:any)=>['c1','c2','c3'].some(key=>r.candidate_id===comparison.value?.[key+'_id'])).map((r:any)=>({...r,criterion:['c1','c2','c3'].filter(key=>r.candidate_id===comparison.value?.[key+'_id']).map(key=>key.toUpperCase()).join(' / ')})))
function fmt(n:any,d:number){return typeof n==='number'&&Number.isFinite(n)?n.toFixed(d):'—'}
</script>
<style scoped>
.profile-validation{margin:20px 0;border-radius:12px;border-color:#dfe7e9}.note{font-size:12px;line-height:1.8;color:#647982}.stats{display:flex;flex-wrap:wrap;gap:18px;font-size:12px;background:#edf6f5;padding:14px;border-radius:8px}.profile-validation :deep(.el-alert){margin:12px 0}
</style>
