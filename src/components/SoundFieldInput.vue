<template>
  <el-card shadow="never" class="sound-field-input">
    <template #header><strong>{{ t('Sound-field model') }}</strong></template>
    <el-radio-group v-model="mode" :disabled="disabled" :aria-label="t('Sound-field model')">
      <el-radio-button value="munk">{{ t('Ideal Munk model') }}</el-radio-button>
      <el-radio-button value="profiles">{{ t('Background + historical profiles') }}</el-radio-button>
    </el-radio-group>
    <p v-if="mode==='munk'" class="note">{{ t('No profile files are required. The default uses a Munk background with the declared analytic perturbations.') }}</p>
    <template v-else>
      <p class="note">{{ t('Upload a measured background and dated historical sound-speed profiles from the same region. Historical minus background defines the perturbation. Only the training period contributes to route selection.') }}</p>
      <div class="actions">
        <el-button :disabled="disabled" :loading="loading" @click="loadExample">{{ t('Load South China Sea example') }}</el-button>
        <el-button :disabled="disabled" @click="downloadExample">{{ t('Download example files') }}</el-button>
      </div>
      <div class="example-guide">
        <strong>{{ t('About these example files') }}</strong>
        <p>{{ t('Load example fills the two inputs on this page. Download example saves a ZIP for inspecting the CSV format or uploading the files manually; downloading does not change the current inputs.') }}</p>
        <p><b>background.csv</b> — {{ t('A fixed background constructed from measured down/up CTD casts and 37 days of 2021 regional data.') }}</p>
        <p><b>history.csv</b> — {{ t('366 daily profiles from 2020 near 18.75°N, 116.50°E, aligned to that background. They retain the 2020–2021 sound-speed differences; they are derived reanalysis profiles, not raw CTD measurements.') }}</p>
        <p><b>{{ t('2020 data source') }}</b> — <a href="https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description" target="_blank" rel="noopener noreferrer">Copernicus Marine · GLORYS12</a>. {{ t('Downloaded from the Copernicus Marine Data Store: Global Ocean Physics Reanalysis, GLOBAL_MULTIYEAR_PHY_001_030; daily dataset cmems_mod_glo_phy_my_0.083deg_P1D-m, version 202311. Potential temperature (thetao) and salinity (so), 2020-01-01 to 2020-12-31, are converted to sound speed with TEOS-10.') }} <a href="https://doi.org/10.48670/moi-00021" target="_blank" rel="noopener noreferrer">DOI: 10.48670/moi-00021</a></p>
        <p>{{ t('The preset uses 182 days for risk modelling, a 14-day gap and 170 held-out days for testing. The ZIP also includes README.md and provenance.json with processing assumptions, attribution and file hashes. A single-day test is selected initially; the full held-out period is optional.') }}</p>
      </div>
      <el-alert v-if="settings.provenance" :closable="false" type="info" :title="t('Example: derived daily reanalysis profiles aligned to a measured background; retrospective time holdout.')"/>
      <fieldset :disabled="disabled || loading">
        <div class="files">
          <label>{{ t('1. Background CSV') }}<input :key="fileInputKey" type="file" accept=".csv" :aria-label="t('Background CSV')" @change="upload($event,'background_csv')"/><small>{{ settings.background_name || (settings.background_csv ? t('Background file loaded') : 'depth_m,sound_speed_mps') }} <span v-if="settings.background_csv">· {{ t(settings.background_source==='example'?'Bundled example':'User upload') }}</span></small></label>
          <label>{{ t('2. Historical CSV') }}<input :key="fileInputKey" type="file" accept=".csv" :aria-label="t('Historical CSV')" @change="upload($event,'history_csv')"/><small>{{ settings.history_name || (settings.history_csv ? t('Historical file loaded') : 'date,depth_m,sound_speed_mps') }} <span v-if="settings.history_csv">· {{ t(settings.history_source==='example'?'Bundled example':'User upload') }}</span></small></label>
        </div>
      </fieldset>
      <p class="note">{{ t('Only the two files shown above are used. Load example replaces both files; a manual upload replaces only its own input. The files are never appended or mixed silently. Check that the background and historical profiles belong to a compatible region and reference convention.') }}</p>
      <el-form label-position="top" :disabled="disabled || loading" class="controls">
        <el-form-item :label="t('Training end (inclusive)')"><el-date-picker v-model="settings.training_end" type="date" value-format="YYYY-MM-DD" :clearable="false" :aria-label="t('Training end (inclusive)')"/></el-form-item>
        <el-form-item :label="t('Validation start (inclusive)')"><el-date-picker v-model="settings.validation_start" type="date" value-format="YYYY-MM-DD" :clearable="false" :aria-label="t('Validation start (inclusive)')"/></el-form-item>
        <el-form-item :label="t('Perturbation multiplier')"><el-select v-model="settings.amplitude" :aria-label="t('Perturbation multiplier')"><el-option v-for="a in [.5,1,2,5]" :key="a" :value="a" :label="`${a}×`"/></el-select></el-form-item>
        <el-form-item :label="t('Validation scope')"><el-select v-model="settings.validation_scope" :aria-label="t('Validation scope')"><el-option value="single" :label="t('One held-out date (quick test)')"/><el-option value="all" :label="t('Every held-out date (longer run)')"/></el-select></el-form-item>
        <el-form-item :label="t('Detailed solution date')"><el-date-picker v-model="settings.validation_date" type="date" value-format="YYYY-MM-DD" :clearable="false" :aria-label="t('Detailed solution date')"/></el-form-item>
        <el-form-item :label="t('Check input files and split')"><el-button :loading="checking" @click="check">{{ t('Check profiles') }}</el-button></el-form-item>
      </el-form>
      <p class="note">{{ t('Dates after training and before validation form an unused gap. Each test profile stays static for the entire survey. Multipliers change amplitude, not spatial gradient or occurrence probability.') }}</p>
      <p class="note">{{ t('CSV only: depth in metres, speed in m/s, dates YYYY-MM-DD. Profiles must cover sea level to at least 100 m below the deepest beacon. All supplied knots are retained; there is no automatic deep extension.') }}</p>
      <div v-if="metadata" class="metadata" role="status">
        <span>{{ t('Training profiles') }} <b>{{ metadata.training_count }}</b></span><span>{{ t('Gap profiles') }} <b>{{ metadata.gap_count }}</b></span><span>{{ t('Held-out profiles') }} <b>{{ metadata.validation_count }}</b></span><span>{{ t('Depth knots') }} <b>{{ metadata.depth_count }}</b></span><span>{{ t('Maximum depth') }} <b>{{ metadata.max_depth_m }} m</b></span>
      </div>
      <p class="note" role="status">{{ t(checking?'Updating file statistics…':'Statistics update automatically when files or validation settings change. Counts refer to profiles actually present in the files.') }}</p>
      <p v-if="error" class="error" role="alert">{{ t(error) }}</p>
    </template>
  </el-card>
</template>
<script setup lang="ts">
import {ref,watch,nextTick,onBeforeUnmount} from 'vue'
import {t} from '../i18n'
import {apiClient} from '../api'
const mode=defineModel<string>('mode',{required:true})
const settings=defineModel<any>('settings',{required:true})
defineProps<{disabled:boolean}>()
const loading=ref(false),checking=ref(false),error=ref(''),metadata=ref<any>(null)
const fileInputKey=ref(0)
let checkVersion=0,checkTimer:ReturnType<typeof setTimeout>|undefined
const message=(e:any)=>typeof e?.response?.data?.detail==='string'?e.response.data.detail:e.message
watch(()=>[settings.value.background_csv,settings.value.history_csv,settings.value.training_end,settings.value.validation_start,settings.value.validation_date,settings.value.validation_scope,settings.value.amplitude],()=>{
  checkVersion++;clearTimeout(checkTimer);metadata.value=null;error.value='';checking.value=false
  if(settings.value.background_csv && settings.value.history_csv)checkTimer=setTimeout(check,350)
},{immediate:true})
async function check(){clearTimeout(checkTimer);const version=++checkVersion;const snapshot={...settings.value};checking.value=true;error.value='';try{const response=await apiClient.post('/api/simulation/profiles/check',snapshot);if(version===checkVersion)metadata.value=response.data}catch(e){if(version===checkVersion)error.value=message(e)}finally{if(version===checkVersion)checking.value=false}}
async function upload(event:Event,key:string){const file=(event.target as HTMLInputElement).files?.[0];if(!file)return;error.value='';if(file.size>16*1024*1024){error.value='Upload both profile CSV files, each at most 16 MiB.';(event.target as HTMLInputElement).value='';return}loading.value=true;try{const kind=key==='background_csv'?'background':'history';settings.value={...settings.value,[key]:await file.text(),[kind+'_name']:file.name,[kind+'_source']:'upload',provenance:null}}catch(e){error.value=message(e)}finally{loading.value=false}}
async function loadExample(){loading.value=true;error.value='';try{const data=(await apiClient.get('/api/simulation/profile-example')).data;settings.value={...data,background_name:'background.csv',history_name:'history.csv',background_source:'example',history_source:'example'};fileInputKey.value++;await nextTick();await check()}catch(e){error.value=message(e)}finally{loading.value=false}}
async function downloadExample(){try{const response=await apiClient.get('/api/simulation/profile-example/download',{responseType:'blob'});const url=URL.createObjectURL(response.data);const a=document.createElement('a');a.href=url;a.download='south-china-sea-profiles.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}catch(e){error.value=message(e)}}
onBeforeUnmount(()=>{checkVersion++;clearTimeout(checkTimer)})
</script>
<style scoped>
.sound-field-input{border-radius:12px;border-color:#dfe7e9;margin-bottom:20px}.note{font-size:12px;color:#647982;line-height:1.8}.actions{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}.actions .el-button{margin:0}.files,.controls{display:grid;grid-template-columns:1fr 1fr;gap:12px 18px}.files label{display:flex;flex-direction:column;gap:10px;font-size:13px;min-width:0}.files input{max-width:100%}.files small{font-size:11px;color:#647982;overflow-wrap:anywhere}fieldset{border:0;margin:16px 0;padding:0}.controls :deep(.el-date-editor),.controls :deep(.el-select){width:100%}.metadata{display:flex;gap:16px;flex-wrap:wrap;background:#edf6f5;padding:12px;border-radius:8px;font-size:12px}.error{color:#a34f39;font-size:13px}@media(max-width:700px){.files,.controls{grid-template-columns:1fr}}
.example-guide{padding:14px 16px;background:#f4f8f9;border:1px solid #e1eaec;border-radius:8px;margin:12px 0;color:#526d77;font-size:12px;line-height:1.75}.example-guide strong{color:#295b65}.example-guide p{margin:7px 0}
</style>
